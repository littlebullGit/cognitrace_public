import Accelerate
import Foundation

struct FFTSpectrum {
    let frequencies: [Float]
    let magnitudes: [Float]
    let powers: [Float]
}

final class FFTProcessor {

    let fftSize: Int

    private let log2n: vDSP_Length
    private let fftSetup: FFTSetup
    private let window: [Float]

    init?(fftSize: Int = 2048) {
        guard fftSize > 0, fftSize.nonzeroBitCount == 1 else {
            return nil
        }

        self.fftSize = fftSize
        self.log2n = vDSP_Length(Int(log2(Double(fftSize))))

        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return nil
        }
        self.fftSetup = fftSetup

        var window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        self.window = window
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    func applyWindow(to signal: [Float]) -> [Float] {
        let padded = paddedSignal(signal)
        var output = [Float](repeating: 0, count: fftSize)
        vDSP_vmul(padded, 1, window, 1, &output, 1, vDSP_Length(fftSize))
        return output
    }

    func powerSpectrum(for signal: [Float], sampleRate: Float) -> FFTSpectrum {
        let windowed = applyWindow(to: signal)

        var real = [Float](repeating: 0, count: fftSize / 2)
        var imag = [Float](repeating: 0, count: fftSize / 2)
        var powers = [Float](repeating: 0, count: fftSize / 2)

        real.withUnsafeMutableBufferPointer { realBuffer in
            imag.withUnsafeMutableBufferPointer { imagBuffer in
                var splitComplex = DSPSplitComplex(
                    realp: realBuffer.baseAddress!,
                    imagp: imagBuffer.baseAddress!
                )

                windowed.withUnsafeBufferPointer { windowedBuffer in
                    let complexPointer = UnsafeRawPointer(windowedBuffer.baseAddress!)
                        .bindMemory(to: DSPComplex.self, capacity: fftSize / 2)
                    vDSP_ctoz(complexPointer, 2, &splitComplex, 1, vDSP_Length(fftSize / 2))
                }

                vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

                splitComplex.imagp[0] = 0

                var scale = Float(1.0 / Float(fftSize))
                vDSP_vsmul(splitComplex.realp, 1, &scale, splitComplex.realp, 1, vDSP_Length(fftSize / 2))
                vDSP_vsmul(splitComplex.imagp, 1, &scale, splitComplex.imagp, 1, vDSP_Length(fftSize / 2))

                vDSP_zvmags(&splitComplex, 1, &powers, 1, vDSP_Length(fftSize / 2))
            }
        }

        let magnitudes = powers.map { Foundation.sqrt(Swift.max(0, $0)) }
        let frequencies = (0..<(fftSize / 2)).map { Float($0) * sampleRate / Float(fftSize) }
        return FFTSpectrum(frequencies: frequencies, magnitudes: magnitudes, powers: powers)
    }

    private func paddedSignal(_ signal: [Float]) -> [Float] {
        if signal.count == fftSize {
            return signal
        }
        if signal.count > fftSize {
            return Array(signal.prefix(fftSize))
        }

        var padded = signal
        padded.append(contentsOf: repeatElement(0, count: fftSize - signal.count))
        return padded
    }
}
