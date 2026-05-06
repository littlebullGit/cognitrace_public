import Accelerate
import Foundation

struct VoiceFeatures {
    let values: [String: Float]
    let trace: [String: Double]

    subscript(_ key: String) -> Float {
        values[key] ?? 0
    }
}

private struct SpectralFrameFeatures {
    let centroid: Float
    let bandwidth: Float
    let rolloff: Float
    let slope: Float
    let alphaRatio: Float
    let hammarbergIndex: Float
    let barkLoudness: Float
}

final class FeatureExtractor {

    static let referenceFeatureNames: [String] = [
        "duration_s",
        "f0_mean", "f0_std", "f0_cv", "f0_min", "f0_max", "f0_range", "f0_p25", "f0_p75", "f0_iqr", "voiced_fraction",
        "jitter_local", "jitter_rap", "jitter_ppq5",
        "shimmer_local", "shimmer_apq3", "shimmer_apq5",
        "hnr",
        "spec_centroid_mean", "spec_centroid_std", "spec_bandwidth_mean", "spec_bandwidth_std",
        "spec_rolloff_mean", "spec_rolloff_std", "spec_slope_mean", "spec_slope_std",
        "mfcc_0_mean", "mfcc_0_std", "mfcc_1_mean", "mfcc_1_std", "mfcc_2_mean", "mfcc_2_std", "mfcc_3_mean", "mfcc_3_std",
        "mfcc_4_mean", "mfcc_4_std", "mfcc_5_mean", "mfcc_5_std", "mfcc_6_mean", "mfcc_6_std", "mfcc_7_mean", "mfcc_7_std",
        "mfcc_8_mean", "mfcc_8_std", "mfcc_9_mean", "mfcc_9_std", "mfcc_10_mean", "mfcc_10_std", "mfcc_11_mean", "mfcc_11_std",
        "mfcc_12_mean", "mfcc_12_std",
        "rms_energy", "log_energy", "zcr", "peak_amplitude",
    ]

    private let fftProcessor: FFTProcessor

    init?(fftSize: Int = 2048) {
        guard let fftProcessor = FFTProcessor(fftSize: fftSize) else {
            return nil
        }
        self.fftProcessor = fftProcessor
    }

    func extract(signal: [Float], sampleRate: Float) -> VoiceFeatures {
        let totalStart = Date().timeIntervalSinceReferenceDate
        var values: [String: Float] = [:]
        var trace: [String: Double] = [:]

        let trimmed = measure("trim_ms", trace: &trace) {
            trimActiveSegment(signal, sampleRate: sampleRate)
        }

        let normalized = measure("normalize_ms", trace: &trace) {
            normalize(trimmed)
        }

        values["duration_s"] = Float(normalized.count) / sampleRate

        let f0Track = measure("f0_track_ms", trace: &trace) {
            computeF0Track(signal: normalized, sampleRate: sampleRate)
        }
        measureVoid("f0_stats_ms", trace: &trace) {
            populateF0Statistics(track: f0Track, values: &values, sampleCount: normalized.count, sampleRate: sampleRate)
        }
        measureVoid("jitter_ms", trace: &trace) {
            populateJitter(track: f0Track, values: &values)
        }
        measureVoid("shimmer_ms", trace: &trace) {
            populateShimmer(signal: normalized, sampleRate: sampleRate, f0Track: f0Track, values: &values)
        }
        values["hnr"] = measure("hnr_ms", trace: &trace) {
            computeHNR(signal: normalized, sampleRate: sampleRate, f0: values["f0_mean"] ?? 0)
        }

        measureVoid("spectral_ms", trace: &trace) {
            populateSpectral(signal: normalized, sampleRate: sampleRate, values: &values)
        }
        measureVoid("mfcc_ms", trace: &trace) {
            populateMFCC(signal: normalized, sampleRate: sampleRate, values: &values)
        }
        measureVoid("energy_ms", trace: &trace) {
            populateEnergy(signal: normalized, values: &values)
        }

        for key in Self.referenceFeatureNames where values[key] == nil {
            values[key] = 0
        }

        trace["feature_total_ms"] = (Date().timeIntervalSinceReferenceDate - totalStart) * 1000

        return VoiceFeatures(values: values, trace: trace)
    }

    /// Trim leading and trailing quiet regions while preserving a small amount
    /// of padding around the detected active span.
    ///
    /// This is intentionally conservative:
    /// - it requires several consecutive active frames before declaring speech
    ///   onset, so a single tap/click does not start the segment
    /// - it keeps padding on both sides to avoid cutting into the task onset
    /// - if no stable active region is found, it returns the original signal
    private func trimActiveSegment(_ signal: [Float], sampleRate: Float) -> [Float] {
        guard !signal.isEmpty else { return signal }

        let frameLength = max(1, Int(0.025 * sampleRate)) // 25 ms
        let hopLength = max(1, Int(0.010 * sampleRate))   // 10 ms
        guard signal.count > frameLength * 2 else { return signal }

        let frameCount = max(1, ((signal.count - frameLength) / hopLength) + 1)
        var frameRms: [Float] = []
        frameRms.reserveCapacity(frameCount)

        for frameIndex in 0..<frameCount {
            let start = frameIndex * hopLength
            let end = start + frameLength
            if end > signal.count { break }

            let frame = Array(signal[start..<end])
            var meanSquare: Float = 0
            frame.withUnsafeBufferPointer { buffer in
                guard let base = buffer.baseAddress else { return }
                vDSP_measqv(base, 1, &meanSquare, vDSP_Length(frame.count))
            }
            frameRms.append(Foundation.sqrt(meanSquare))
        }

        guard let peakRms = frameRms.max(), peakRms > Float.ulpOfOne else {
            return signal
        }

        let energyFloor: Float = 0.003
        let activeThreshold = max(energyFloor, peakRms * 0.12)
        let onsetFramesRequired = 4
        let offsetFramesRequired = 8

        var startFrame: Int?
        var activeRun = 0
        for index in frameRms.indices {
            if frameRms[index] >= activeThreshold {
                activeRun += 1
                if startFrame == nil, activeRun >= onsetFramesRequired {
                    startFrame = max(0, index - onsetFramesRequired + 1)
                    break
                }
            } else {
                activeRun = 0
            }
        }

        guard let detectedStartFrame = startFrame else { return signal }

        var endFrame: Int?
        var trailingQuietDetected = false
        var reverseActiveRun = 0
        for index in stride(from: frameRms.count - 1, through: detectedStartFrame, by: -1) {
            if frameRms[index] < activeThreshold {
                if !trailingQuietDetected {
                    trailingQuietDetected = true
                }
                reverseActiveRun = 0
            } else {
                reverseActiveRun += 1
                if reverseActiveRun >= offsetFramesRequired {
                    endFrame = min(frameRms.count - 1, index + offsetFramesRequired - 1)
                    break
                }
            }
        }

        let resolvedEndFrame: Int
        if let endFrame {
            resolvedEndFrame = endFrame
        } else if trailingQuietDetected {
            // We saw trailing quiet frames but could not find a long enough
            // active run before them, so end at the first active frame after
            // the quiet tail instead of returning the full signal.
            resolvedEndFrame = detectedStartFrame
        } else {
            resolvedEndFrame = frameRms.count - 1
        }

        let padFrames = 10 // ~100 ms
        let startSample = max(0, (max(0, detectedStartFrame - padFrames) * hopLength))
        let endSample = min(
            signal.count,
            ((min(frameRms.count - 1, resolvedEndFrame + padFrames) * hopLength) + frameLength)
        )

        guard endSample > startSample, endSample - startSample >= Int(0.35 * sampleRate) else {
            return signal
        }

        return Array(signal[startSample..<endSample])
    }

    private func normalize(_ signal: [Float]) -> [Float] {
        guard !signal.isEmpty else { return signal }

        var maxValue: Float = 0
        signal.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            vDSP_maxmgv(baseAddress, 1, &maxValue, vDSP_Length(signal.count))
        }

        guard maxValue > 0 else {
            return signal
        }

        var divisor = maxValue
        var normalized = [Float](repeating: 0, count: signal.count)
        signal.withUnsafeBufferPointer { source in
            normalized.withUnsafeMutableBufferPointer { destination in
                guard let sourceBase = source.baseAddress,
                      let destinationBase = destination.baseAddress else { return }
                vDSP_vsdiv(sourceBase, 1, &divisor, destinationBase, 1, vDSP_Length(signal.count))
            }
        }
        return normalized
    }

    private func computeF0Track(signal: [Float], sampleRate: Float, frameLength: Float = 0.03, hop: Float = 0.01, fmin: Float = 50, fmax: Float = 500) -> [Float] {
        let frameSamples = max(1, Int(frameLength * sampleRate))
        let hopSamples = max(1, Int(hop * sampleRate))
        let frameCount = max(1, ((signal.count - frameSamples) / hopSamples) + 1)

        var track: [Float] = []
        track.reserveCapacity(frameCount)

        signal.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            for frameIndex in 0..<frameCount {
                let start = frameIndex * hopSamples
                let end = start + frameSamples
                if end > signal.count { break }
                let f0 = computeF0Autocorr(
                    baseAddress: baseAddress.advanced(by: start),
                    count: frameSamples,
                    sampleRate: sampleRate,
                    fmin: fmin,
                    fmax: fmax
                )
                if f0 > 0 {
                    track.append(f0)
                }
            }
        }

        return track
    }

    private func computeF0Autocorr(baseAddress: UnsafePointer<Float>, count: Int, sampleRate: Float, fmin: Float, fmax: Float) -> Float {
        let minLag = max(1, Int(sampleRate / fmax))
        let maxLag = min(count - 1, Int(sampleRate / fmin))
        if count < maxLag || minLag >= maxLag { return 0 }

        var mean: Float = 0
        vDSP_meanv(baseAddress, 1, &mean, vDSP_Length(count))
        var negativeMean = -mean
        var centered = [Float](repeating: 0, count: count)
        centered.withUnsafeMutableBufferPointer { centeredBuffer in
            guard let centeredBase = centeredBuffer.baseAddress else { return }
            vDSP_vsadd(baseAddress, 1, &negativeMean, centeredBase, 1, vDSP_Length(count))
        }
        let energy = centered.reduce(0) { $0 + ($1 * $1) }
        if energy < 1e-10 { return 0 }

        var bestLag = 0
        var bestCorrelation: Float = 0

        centered.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            for lag in minLag...maxLag {
                var sum: Float = 0
                vDSP_dotpr(
                    baseAddress,
                    1,
                    baseAddress.advanced(by: lag),
                    1,
                    &sum,
                    vDSP_Length(count - lag)
                )
                let correlation = sum / energy
                if correlation > bestCorrelation {
                    bestCorrelation = correlation
                    bestLag = lag
                }
            }
        }

        guard bestLag > 0, bestCorrelation > 0.2 else { return 0 }
        return sampleRate / Float(bestLag)
    }

    private func populateF0Statistics(track: [Float], values: inout [String: Float], sampleCount: Int, sampleRate: Float) {
        guard !track.isEmpty else {
            ["f0_mean", "f0_std", "f0_cv", "f0_min", "f0_max", "f0_range", "f0_p25", "f0_p75", "f0_iqr", "voiced_fraction"].forEach {
                values[$0] = 0
            }
            return
        }

        let mean = track.mean
        let std = track.standardDeviation
        let p25 = percentile(track, percentile: 0.25)
        let p75 = percentile(track, percentile: 0.75)

        values["f0_mean"] = mean
        values["f0_std"] = std
        values["f0_cv"] = safeDivide(std, mean)
        values["f0_min"] = track.min() ?? 0
        values["f0_max"] = track.max() ?? 0
        values["f0_range"] = (track.max() ?? 0) - (track.min() ?? 0)
        values["f0_p25"] = p25
        values["f0_p75"] = p75
        values["f0_iqr"] = p75 - p25
        let denominator = max(1, sampleCount / Int(0.01 * sampleRate))
        values["voiced_fraction"] = Float(track.count) / Float(denominator)
    }

    private func populateJitter(track: [Float], values: inout [String: Float]) {
        guard track.count >= 3 else {
            values["jitter_local"] = 0
            values["jitter_rap"] = 0
            values["jitter_ppq5"] = 0
            return
        }

        let periods = track.map { 1 / $0 }
        let meanPeriod = periods.mean
        let diffs = zip(periods.dropFirst(), periods).map { abs($0 - $1) }
        values["jitter_local"] = safeDivide(diffs.mean, meanPeriod)

        if periods.count >= 3 {
            let rap = (1..<(periods.count - 1)).map { idx in
                let average = (periods[idx - 1] + periods[idx] + periods[idx + 1]) / 3
                return abs(periods[idx] - average)
            }
            values["jitter_rap"] = safeDivide(rap.mean, meanPeriod)
        } else {
            values["jitter_rap"] = 0
        }

        if periods.count >= 5 {
            let ppq = (2..<(periods.count - 2)).map { idx in
                let average = Array(periods[(idx - 2)...(idx + 2)]).mean
                return abs(periods[idx] - average)
            }
            values["jitter_ppq5"] = safeDivide(ppq.mean, meanPeriod)
        } else {
            values["jitter_ppq5"] = 0
        }
    }

    private func populateShimmer(signal: [Float], sampleRate: Float, f0Track: [Float], values: inout [String: Float]) {
        guard f0Track.count >= 3 else {
            values["shimmer_local"] = 0
            values["shimmer_apq3"] = 0
            values["shimmer_apq5"] = 0
            return
        }

        let periods = f0Track.map { 1 / $0 }
        let frameLength = Int(periods.mean * sampleRate)
        guard frameLength >= 10 else {
            values["shimmer_local"] = 0
            values["shimmer_apq3"] = 0
            values["shimmer_apq5"] = 0
            return
        }

        var amplitudes: [Float] = []
        amplitudes.reserveCapacity(f0Track.count)
        var index = 0
        signal.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            while index + frameLength <= signal.count && amplitudes.count < f0Track.count {
                var peak: Float = 0
                vDSP_maxmgv(
                    baseAddress.advanced(by: index),
                    1,
                    &peak,
                    vDSP_Length(frameLength)
                )
                amplitudes.append(peak)
                index += frameLength
            }
        }

        guard amplitudes.count >= 3 else {
            values["shimmer_local"] = 0
            values["shimmer_apq3"] = 0
            values["shimmer_apq5"] = 0
            return
        }

        let meanAmplitude = amplitudes.mean
        let diffs = zip(amplitudes.dropFirst(), amplitudes).map { abs($0 - $1) }
        values["shimmer_local"] = safeDivide(diffs.mean, meanAmplitude)

        if amplitudes.count >= 3 {
            let apq3 = (1..<(amplitudes.count - 1)).map { idx in
                let average = (amplitudes[idx - 1] + amplitudes[idx] + amplitudes[idx + 1]) / 3
                return abs(amplitudes[idx] - average)
            }
            values["shimmer_apq3"] = safeDivide(apq3.mean, meanAmplitude)
        } else {
            values["shimmer_apq3"] = 0
        }

        if amplitudes.count >= 5 {
            let apq5 = (2..<(amplitudes.count - 2)).map { idx in
                let average = Array(amplitudes[(idx - 2)...(idx + 2)]).mean
                return abs(amplitudes[idx] - average)
            }
            values["shimmer_apq5"] = safeDivide(apq5.mean, meanAmplitude)
        } else {
            values["shimmer_apq5"] = 0
        }
    }

    /// Compute Harmonics-to-Noise Ratio using per-frame autocorrelation.
    ///
    /// Processes voiced frames only (autocorrelation peak > 0.3), skipping
    /// unvoiced segments (consonants, silence, breathing). This matches
    /// Praat's approach and gives meaningful HNR for multi-task recordings.
    private func computeHNR(signal: [Float], sampleRate: Float, f0: Float) -> Float {
        guard f0 > 0 else { return 0 }

        let frameSamples = Int(0.03 * sampleRate) // 30 ms frames
        let hopSamples   = Int(0.01 * sampleRate) // 10 ms hops
        guard frameSamples > 0, signal.count >= frameSamples else { return 0 }

        let frameCount = (signal.count - frameSamples) / hopSamples + 1
        let minLag = max(1, Int(sampleRate / 500))   // fmax 500 Hz
        let maxLag = min(frameSamples - 1, Int(sampleRate / 50)) // fmin 50 Hz
        guard minLag < maxLag else { return 0 }

        var hnrSum: Float = 0
        var voicedCount   = 0

        signal.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }

            for i in 0..<frameCount {
                let start = i * hopSamples
                guard start + frameSamples <= signal.count else { break }
                let framePtr = base.advanced(by: start)

                // Center the frame (subtract mean).
                var mean: Float = 0
                vDSP_meanv(framePtr, 1, &mean, vDSP_Length(frameSamples))
                var negMean = -mean
                var centered = [Float](repeating: 0, count: frameSamples)
                centered.withUnsafeMutableBufferPointer { buf in
                    guard let dst = buf.baseAddress else { return }
                    vDSP_vsadd(framePtr, 1, &negMean, dst, 1, vDSP_Length(frameSamples))
                }

                // Frame energy.
                let energy: Float = centered.withUnsafeBufferPointer { buf in
                    guard let p = buf.baseAddress else { return 0 }
                    var e: Float = 0
                    vDSP_dotpr(p, 1, p, 1, &e, vDSP_Length(frameSamples))
                    return e
                }
                guard energy > 1e-10 else { continue }

                // Best normalized autocorrelation in the pitch-period range.
                var bestR: Float = 0
                centered.withUnsafeBufferPointer { buf in
                    guard let p = buf.baseAddress else { return }
                    for lag in minLag...maxLag {
                        var sum: Float = 0
                        vDSP_dotpr(p, 1, p.advanced(by: lag), 1, &sum,
                                   vDSP_Length(frameSamples - lag))
                        let r = sum / energy
                        if r > bestR { bestR = r }
                    }
                }

                // Skip unvoiced frames.
                guard bestR > 0.3 else { continue }

                // HNR = 10 · log10( r / (1 − r) )
                let hnr = 10 * log10(max(1e-10, bestR / max(1e-10, 1 - bestR)))
                hnrSum += hnr
                voicedCount += 1
            }
        }

        guard voicedCount > 0 else { return 0 }
        return min(40, max(-20, hnrSum / Float(voicedCount)))
    }

    private func populateSpectral(signal: [Float], sampleRate: Float, values: inout [String: Float]) {
        let frameLength = min(fftProcessor.fftSize, signal.count)
        guard frameLength > 0 else { return }

        let hop = max(1, frameLength / 2)
        let frameCount = max(1, ((signal.count - frameLength) / hop) + 1)
        var centroidValues: [Float] = []
        var bandwidthValues: [Float] = []
        var rolloffValues: [Float] = []
        var slopeValues: [Float] = []
        var alphaRatioValues: [Float] = []
        var hammarbergValues: [Float] = []
        var barkLoudnessValues: [Float] = []
        var fluxValues: [Float] = []
        var previousNormalizedMagnitudes: [Float]?

        for frameIndex in 0..<min(frameCount, 100) {
            let start = frameIndex * hop
            let end = start + frameLength
            if end > signal.count { break }

            let frame = Array(signal[start..<end])
            let spectrum = fftProcessor.powerSpectrum(for: frame, sampleRate: sampleRate)
            let magnitudes = spectrum.magnitudes
            let frequencies = spectrum.frequencies
            guard let frameFeatures = computeSpectralFrameFeatures(
                magnitudes: magnitudes,
                powers: spectrum.powers,
                frequencies: frequencies
            ) else {
                continue
            }

            centroidValues.append(frameFeatures.centroid)
            bandwidthValues.append(frameFeatures.bandwidth)
            rolloffValues.append(frameFeatures.rolloff)
            slopeValues.append(frameFeatures.slope)
            alphaRatioValues.append(frameFeatures.alphaRatio)
            hammarbergValues.append(frameFeatures.hammarbergIndex)
            barkLoudnessValues.append(frameFeatures.barkLoudness)

            let normalizedMagnitudes = normalizeVector(magnitudes)
            if let previousNormalizedMagnitudes {
                fluxValues.append(spectralFlux(current: normalizedMagnitudes, previous: previousNormalizedMagnitudes))
            }
            previousNormalizedMagnitudes = normalizedMagnitudes
        }

        values["spec_centroid_mean"] = centroidValues.mean
        values["spec_centroid_std"] = centroidValues.standardDeviation
        values["spec_bandwidth_mean"] = bandwidthValues.mean
        values["spec_bandwidth_std"] = bandwidthValues.standardDeviation
        values["spec_rolloff_mean"] = rolloffValues.mean
        values["spec_rolloff_std"] = rolloffValues.standardDeviation
        values["spec_slope_mean"] = slopeValues.mean
        values["spec_slope_std"] = slopeValues.standardDeviation
        values["spec_alpha_ratio_mean"] = alphaRatioValues.mean
        values["spec_alpha_ratio_std"] = alphaRatioValues.standardDeviation
        values["spec_hammarberg_mean"] = hammarbergValues.mean
        values["spec_hammarberg_std"] = hammarbergValues.standardDeviation
        values["spec_flux_mean"] = fluxValues.mean
        values["spec_flux_std"] = fluxValues.standardDeviation
        values["loudness_bark_mean"] = barkLoudnessValues.mean
        values["loudness_bark_std"] = barkLoudnessValues.standardDeviation
    }

    private func populateMFCC(signal: [Float], sampleRate: Float, values: inout [String: Float], coefficientCount: Int = 13, melBands: Int = 40) {
        let emphasized = preEmphasis(signal)
        let frameLength = fftProcessor.fftSize
        let hop = max(1, frameLength / 2)
        let frameCount = max(1, ((emphasized.count - frameLength) / hop) + 1)
        let filterBank = makeMelFilterBank(sampleRate: sampleRate, melBandCount: melBands)
        let dctMatrix = makeDCTMatrix(coefficientCount: coefficientCount, inputCount: melBands)
        var coefficientSums = [Float](repeating: 0, count: coefficientCount)
        var coefficientSquaredSums = [Float](repeating: 0, count: coefficientCount)
        var processedFrameCount = 0

        for frameIndex in 0..<frameCount {
            let start = frameIndex * hop
            let end = min(start + frameLength, emphasized.count)
            var frame = Array(emphasized[start..<end])
            if frame.count < frameLength {
                frame.append(contentsOf: repeatElement(0, count: frameLength - frame.count))
            }

            let spectrum = fftProcessor.powerSpectrum(for: frame, sampleRate: sampleRate)
            let melEnergies: [Float] = filterBank.map { filter in
                var sum: Float = 0
                let count = Swift.min(filter.count, spectrum.powers.count)
                for index in 0..<count {
                    sum += filter[index] * spectrum.powers[index]
                }
                return Foundation.log(Swift.max(sum, Float.ulpOfOne))
            }

            var coefficients = [Float](repeating: 0, count: coefficientCount)
            for coeffIndex in 0..<coefficientCount {
                var value: Float = 0
                let count = Swift.min(dctMatrix[coeffIndex].count, melEnergies.count)
                for index in 0..<count {
                    value += dctMatrix[coeffIndex][index] * melEnergies[index]
                }
                coefficients[coeffIndex] = value
                coefficientSums[coeffIndex] += value
                coefficientSquaredSums[coeffIndex] += value * value
            }
            processedFrameCount += 1
        }

        let denominator = Float(max(1, processedFrameCount))
        for coefficientIndex in 0..<coefficientCount {
            let mean = coefficientSums[coefficientIndex] / denominator
            let meanSquare = coefficientSquaredSums[coefficientIndex] / denominator
            let variance = Swift.max(0, meanSquare - (mean * mean))
            values["mfcc_\(coefficientIndex)_mean"] = mean
            values["mfcc_\(coefficientIndex)_std"] = Foundation.sqrt(variance)
        }
    }

    private func populateEnergy(signal: [Float], values: inout [String: Float]) {
        guard !signal.isEmpty else {
            values["rms_energy"] = 0
            values["log_energy"] = 0
            values["zcr"] = 0
            values["peak_amplitude"] = 0
            return
        }

        var meanSquare: Float = 0
        var totalEnergy: Float = 0
        var peakAmplitude: Float = 0
        signal.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            vDSP_measqv(baseAddress, 1, &meanSquare, vDSP_Length(signal.count))
            vDSP_svesq(baseAddress, 1, &totalEnergy, vDSP_Length(signal.count))
            vDSP_maxmgv(baseAddress, 1, &peakAmplitude, vDSP_Length(signal.count))
        }

        values["rms_energy"] = Foundation.sqrt(meanSquare)
        values["log_energy"] = Foundation.log(totalEnergy + 1e-10)
        values["peak_amplitude"] = peakAmplitude

        var zeroCrossings = 0
        for index in 1..<signal.count {
            let previous = signal[index - 1]
            let current = signal[index]
            if (previous >= 0 && current < 0) || (previous < 0 && current >= 0) {
                zeroCrossings += 1
            }
        }
        values["zcr"] = Float(zeroCrossings) / Float(max(1, 2 * signal.count))
    }

    private func computeSpectralFrameFeatures(magnitudes: [Float], powers: [Float], frequencies: [Float]) -> SpectralFrameFeatures? {
        let total = magnitudes.reduce(0, +)
        if total < 1e-10 { return nil }

        var weightedFrequencySum: Float = 0
        for index in 0..<Swift.min(frequencies.count, magnitudes.count) {
            let frequency = frequencies[index]
            let magnitude = magnitudes[index]
            weightedFrequencySum += frequency * magnitude
        }
        let centroid = weightedFrequencySum / total

        var bandwidthSum: Float = 0
        var cumulative: Float = 0
        var rolloff: Float = 0
        var lowBandEnergy: Float = 0
        var highBandEnergy: Float = 0
        var lowPeak: Float = 0
        var highPeak: Float = 0
        var barkLoudness: Float = 0
        let target = total * 0.85

        for index in 0..<Swift.min(frequencies.count, magnitudes.count) {
            let frequency = frequencies[index]
            let magnitude = magnitudes[index]
            let power = powers[index]
            let delta = frequency - centroid
            bandwidthSum += (delta * delta) * magnitude

            cumulative += magnitude
            if rolloff == 0, cumulative >= target {
                rolloff = frequency
            }

            if frequency >= 50, frequency < 1000 {
                lowBandEnergy += power
            } else if frequency >= 1000, frequency <= 5000 {
                highBandEnergy += power
            }

            if frequency <= 2000 {
                lowPeak = Swift.max(lowPeak, magnitude)
            } else if frequency <= 5000 {
                highPeak = Swift.max(highPeak, magnitude)
            }

            barkLoudness += barkWeight(frequency: frequency) * Foundation.sqrt(Swift.max(power, 0))
        }

        let bandwidth = Foundation.sqrt(bandwidthSum / total)
        let logMagnitudes = magnitudes.map { Foundation.log10(Swift.max($0, 1e-10)) }
        let slope = linearSlope(x: frequencies, y: logMagnitudes)

        return SpectralFrameFeatures(
            centroid: centroid,
            bandwidth: bandwidth,
            rolloff: rolloff,
            slope: slope,
            alphaRatio: safeDivide(lowBandEnergy, highBandEnergy),
            hammarbergIndex: safeDivide(lowPeak, highPeak),
            barkLoudness: barkLoudness
        )
    }

    private func spectralFlux(current: [Float], previous: [Float]) -> Float {
        let count = Swift.min(current.count, previous.count)
        if count == 0 { return 0 }

        var sum: Float = 0
        for index in 0..<count {
            let delta = current[index] - previous[index]
            sum += delta * delta
        }
        return Foundation.sqrt(sum / Float(count))
    }

    private func normalizeVector(_ values: [Float]) -> [Float] {
        let norm = Foundation.sqrt(values.reduce(Float(0)) { partial, value in
            partial + (value * value)
        })
        if norm <= Float.ulpOfOne {
            return values
        }
        return values.map { $0 / norm }
    }

    private func barkWeight(frequency: Float) -> Float {
        let bark = 13 * atan(0.00076 * Double(frequency)) + 3.5 * atan(pow(Double(frequency) / 7500, 2))
        return Float(1 + (bark / 24))
    }

    private func preEmphasis(_ signal: [Float], factor: Float = 0.97) -> [Float] {
        guard !signal.isEmpty else { return [] }
        var output = [signal[0]]
        output.reserveCapacity(signal.count)
        for index in 1..<signal.count {
            output.append(signal[index] - (factor * signal[index - 1]))
        }
        return output
    }

    private func makeMelFilterBank(sampleRate: Float, melBandCount: Int) -> [[Float]] {
        let binCount = fftProcessor.fftSize / 2
        let minMel: Float = 0
        let maxMel = hzToMel(sampleRate / 2)
        let melPoints = (0..<(melBandCount + 2)).map { index in
            minMel + (Float(index) * (maxMel - minMel) / Float(melBandCount + 1))
        }
        let hzPoints = melPoints.map(melToHz)
        let bins = hzPoints.map { Int(floor(Float(fftProcessor.fftSize + 1) * $0 / sampleRate)) }

        var bank = Array(repeating: [Float](repeating: 0, count: binCount), count: melBandCount)
        for band in 1...melBandCount {
            let left = min(max(0, bins[band - 1]), binCount - 1)
            let center = min(max(0, bins[band]), binCount - 1)
            let right = min(max(0, bins[band + 1]), binCount - 1)

            if center > left {
                for index in left..<center {
                    bank[band - 1][index] = (Float(index) - Float(left)) / (Float(center) - Float(left))
                }
            }
            if right > center {
                for index in center..<right {
                    bank[band - 1][index] = (Float(right) - Float(index)) / (Float(right) - Float(center))
                }
            }
        }

        return bank
    }

    private func makeDCTMatrix(coefficientCount: Int, inputCount: Int) -> [[Float]] {
        (0..<coefficientCount).map { i in
            (0..<inputCount).map { j in
                Foundation.cos(Float.pi * Float(i) * Float((2 * j) + 1) / Float(2 * inputCount))
            }
        }
    }

    private func hzToMel(_ hz: Float) -> Float {
        2595 * Foundation.log10(1 + (hz / 700))
    }

    private func melToHz(_ mel: Float) -> Float {
        700 * (Foundation.pow(10, mel / 2595) - 1)
    }

    private func percentile(_ values: [Float], percentile: Float) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let position = max(0, min(Float(sorted.count - 1), percentile * Float(sorted.count - 1)))
        let lower = Int(floor(position))
        let upper = Int(ceil(position))
        if lower == upper { return sorted[lower] }
        let weight = position - Float(lower)
        return sorted[lower] + ((sorted[upper] - sorted[lower]) * weight)
    }

    private func linearSlope(x: [Float], y: [Float]) -> Float {
        guard x.count == y.count, x.count > 1 else { return 0 }
        let xMean = x.mean
        let yMean = y.mean
        var numerator: Float = 0
        var denominator: Float = 0
        for index in x.indices {
            let xDelta = x[index] - xMean
            numerator += xDelta * (y[index] - yMean)
            denominator += xDelta * xDelta
        }
        return safeDivide(numerator, denominator)
    }

    private func safeDivide(_ numerator: Float, _ denominator: Float, defaultValue: Float = 0) -> Float {
        abs(denominator) > Float.ulpOfOne ? numerator / denominator : defaultValue
    }

    private func measure<T>(_ key: String, trace: inout [String: Double], _ work: () -> T) -> T {
        let start = Date().timeIntervalSinceReferenceDate
        let result = work()
        trace[key] = (Date().timeIntervalSinceReferenceDate - start) * 1000
        return result
    }

    private func measureVoid(_ key: String, trace: inout [String: Double], _ work: () -> Void) {
        let start = Date().timeIntervalSinceReferenceDate
        work()
        trace[key] = (Date().timeIntervalSinceReferenceDate - start) * 1000
    }
}

private extension Array where Element == Float {
    var mean: Float {
        guard !isEmpty else { return 0 }
        return reduce(0, +) / Float(count)
    }

    var standardDeviation: Float {
        guard count > 1 else { return 0 }
        let avg = mean
        let variance = reduce(Float(0)) { partial, value in
            let delta = value - avg
            return partial + (delta * delta)
        } / Float(count)
        return Foundation.sqrt(Swift.max(0, variance))
    }
}
