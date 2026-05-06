import AVFoundation
import Foundation

struct ManifestRow {
    let audioPath: String
    let label: String
    let subject: String
}

@main
struct SwiftFeatureBatch {
    static func main() throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count == 2 else {
            FileHandle.standardError.write(Data("Usage: swift_feature_batch <manifest.csv> <output.csv>\n".utf8))
            Foundation.exit(1)
        }

        let manifestURL = URL(fileURLWithPath: arguments[0])
        let outputURL = URL(fileURLWithPath: arguments[1])
        let rows = try loadManifest(from: manifestURL)
        guard let extractor = FeatureExtractor() else {
            throw NSError(domain: "SwiftFeatureBatch", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not initialize FeatureExtractor"]) }

        let featureNames = FeatureExtractor.referenceFeatureNames
        let traceNames = [
            "normalize_ms",
            "f0_track_ms",
            "f0_stats_ms",
            "jitter_ms",
            "shimmer_ms",
            "hnr_ms",
            "spectral_ms",
            "mfcc_ms",
            "energy_ms",
            "feature_total_ms"
        ]

        var lines: [String] = []
        lines.append((["audio_path", "label", "subject", "sample_rate"] + traceNames + featureNames).joined(separator: ","))

        for row in rows {
            let payload = try loadMono16kPCM(url: URL(fileURLWithPath: row.audioPath))
            let extracted = extractor.extract(signal: payload.samples, sampleRate: payload.sampleRate)
            let traceValues = traceNames.map { String(extracted.trace[$0] ?? 0) }
            let featureValues = featureNames.map { String(extracted[$0]) }
            let csvRow = [row.audioPath, row.label, row.subject, String(payload.sampleRate)] + traceValues + featureValues
            lines.append(csvRow.map(csvEscape).joined(separator: ","))
        }

        try lines.joined(separator: "\n").write(to: outputURL, atomically: true, encoding: .utf8)
    }

    private static func loadManifest(from url: URL) throws -> [ManifestRow] {
        let raw = try String(contentsOf: url, encoding: .utf8)
        let lines = raw.split(whereSeparator: \ .isNewline)
        guard lines.count >= 2 else { return [] }

        return lines.dropFirst().compactMap { line in
            let fields = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            guard fields.count >= 3 else { return nil }
            return ManifestRow(audioPath: fields[0], label: fields[1], subject: fields[2])
        }
    }

    private static func loadMono16kPCM(url: URL) throws -> (samples: [Float], sampleRate: Float) {
        let file = try AVAudioFile(forReading: url)
        let inputFormat = file.processingFormat
        guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                               sampleRate: 16_000,
                                               channels: 1,
                                               interleaved: false),
              let converter = AVAudioConverter(from: inputFormat, to: targetFormat),
              let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw NSError(domain: "SwiftFeatureBatch", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not initialize audio conversion"]) }

        try file.read(into: inputBuffer)

        let ratio = targetFormat.sampleRate / inputFormat.sampleRate
        let outputCapacity = AVAudioFrameCount(ceil(Double(inputBuffer.frameLength) * ratio) + 1)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputCapacity) else {
            throw NSError(domain: "SwiftFeatureBatch", code: 3, userInfo: [NSLocalizedDescriptionKey: "Could not create output buffer"]) }

        var consumed = false
        var error: NSError?
        let status = converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if status == .error {
            throw error ?? NSError(domain: "SwiftFeatureBatch", code: 4, userInfo: [NSLocalizedDescriptionKey: "Audio conversion failed"]) }

        guard let channelData = outputBuffer.floatChannelData?[0] else {
            throw NSError(domain: "SwiftFeatureBatch", code: 5, userInfo: [NSLocalizedDescriptionKey: "Missing output channel data"]) }

        let samples = Array(UnsafeBufferPointer(start: channelData, count: Int(outputBuffer.frameLength)))
        return (samples, Float(targetFormat.sampleRate))
    }

    private static func csvEscape(_ value: String) -> String {
        if value.contains(",") || value.contains("\"") || value.contains("\n") {
            return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        return value
    }
}
