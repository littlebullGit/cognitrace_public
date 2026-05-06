import Accelerate
import AVFoundation
import Flutter

// MARK: - AudioBridge

@objc class AudioBridge: NSObject {
    private let featureExtractionQueue = DispatchQueue(label: "com.cognitrace.feature-extraction", qos: .userInitiated)

    // MARK: - Constants

    private static let channelName = "com.cognitrace/audio"
    private let targetSampleRate: Double = 16_000

    // MARK: - Properties

    private let channel: FlutterMethodChannel
    private var engine: AVAudioEngine?
    private let featureExtractor = FeatureExtractor()
    private var converter: AVAudioConverter?
    private var audioPlayer: AVAudioPlayer?
    private var currentAudioPath: String?

    private let bufferLock = NSLock()
    private var pcmBuffer: [Float] = []

    // Written from audio thread, read from main thread.
    // Torn reads are acceptable — this value drives UI only.
    private var currentRMS: Float = 0.0

    private let stateLock = NSLock()
    private var _isRecording = false
    private var isRecording: Bool {
        get { stateLock.lock(); defer { stateLock.unlock() }; return _isRecording }
        set { stateLock.lock(); defer { stateLock.unlock() }; _isRecording = newValue }
    }

    // MARK: - Init

    @objc init(messenger: FlutterBinaryMessenger) {
        channel = FlutterMethodChannel(name: AudioBridge.channelName, binaryMessenger: messenger)
        super.init()
        channel.setMethodCallHandler(handle(_:result:))
    }

    // MARK: - Method dispatch

    private func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "startRecording":  startRecording(result: result)
        case "stopRecording":   stopRecording(result: result)
        case "getAudioLevel":   result(Double(currentRMS))
        case "extractFeatures": extractFeatures(call: call, result: result)
        case "extractFeaturesFromWavBytes": extractFeaturesFromWavBytes(call: call, result: result)
        case "playAudioFile": playAudioFile(call: call, result: result)
        case "pauseAudioPlayback": pauseAudioPlayback(result: result)
        case "stopAudioPlayback": stopAudioPlayback(result: result)
        case "restartAudioPlayback": restartAudioPlayback(call: call, result: result)
        default:                result(FlutterMethodNotImplemented)
        }
    }

    // MARK: - startRecording

    private func startRecording(result: @escaping FlutterResult) {
        guard !isRecording else {
            result(FlutterError(code: "ALREADY_RECORDING",
                                message: "Call stopRecording before starting a new session.",
                                details: nil))
            return
        }
        AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
            guard let self else { return }
            DispatchQueue.main.async {
                if granted { self.doStart(result: result) }
                else {
                    result(FlutterError(
                        code: "PERMISSION_DENIED",
                        message: "Microphone access denied. Enable it in Settings → Privacy → Microphone.",
                        details: nil))
                }
            }
        }
    }

    private func doStart(result: @escaping FlutterResult) {
        audioPlayer?.stop()
        audioPlayer = nil
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine?.reset()
        engine = nil

        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            // Preferred rate is a hint; hardware may still deliver 44100/48000 Hz.
            // AVAudioConverter will resample from whatever the hardware actually provides.
            try session.setPreferredSampleRate(targetSampleRate)
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            result(FlutterError(code: "AUDIO_SESSION_ERROR",
                                message: "Audio session setup failed: \(error.localizedDescription)",
                                details: nil))
            return
        }

        let engine = AVAudioEngine()
        self.engine = engine
        let inputNode = engine.inputNode
        let hardwareFormat = inputNode.outputFormat(forBus: 0)

        guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                               sampleRate: targetSampleRate,
                                               channels: 1,
                                               interleaved: false) else {
            result(FlutterError(code: "FORMAT_ERROR",
                                message: "Could not create 16 kHz mono format.", details: nil))
            return
        }

        // AVAudioConverter handles sample-rate conversion and stereo→mono down-mix.
        guard let conv = AVAudioConverter(from: hardwareFormat, to: targetFormat) else {
            result(FlutterError(code: "CONVERTER_ERROR",
                                message: "Cannot bridge \(Int(hardwareFormat.sampleRate)) Hz → \(Int(targetSampleRate)) Hz.",
                                details: nil))
            self.engine = nil
            return
        }
        converter = conv

        bufferLock.lock()
        pcmBuffer.removeAll(keepingCapacity: true)
        bufferLock.unlock()
        currentRMS = 0.0
        isRecording = true

        let tapFrames = AVAudioFrameCount(hardwareFormat.sampleRate * 0.1) // ~100 ms per chunk
        inputNode.installTap(onBus: 0, bufferSize: tapFrames, format: hardwareFormat) {
            [weak self] buffer, _ in self?.processTap(buffer)
        }

        do {
            engine.prepare()
            try engine.start()
            result(nil) // nil = success on Flutter MethodChannel
        } catch {
            inputNode.removeTap(onBus: 0)
            isRecording = false
            converter = nil
            self.engine = nil
            result(FlutterError(code: "ENGINE_START_ERROR",
                                message: "AVAudioEngine failed to start: \(error.localizedDescription)",
                                details: nil))
        }
    }

    // MARK: - Tap handler (audio thread)

    private func processTap(_ inputBuffer: AVAudioPCMBuffer) {
        guard let conv = converter else { return }

        let ratio = conv.outputFormat.sampleRate / conv.inputFormat.sampleRate
        let outCapacity = AVAudioFrameCount(ceil(Double(inputBuffer.frameLength) * ratio) + 1)

        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: conv.outputFormat,
                                               frameCapacity: outCapacity) else { return }

        // The converter callback must supply the input buffer exactly once per call.
        var consumed = false
        var convError: NSError?
        let status = conv.convert(to: outBuffer, error: &convError) { _, outStatus in
            if consumed { outStatus.pointee = .noDataNow; return nil }
            outStatus.pointee = .haveData
            consumed = true
            return inputBuffer
        }

        guard status != .error, let channelData = outBuffer.floatChannelData?[0] else { return }
        let frameCount = Int(outBuffer.frameLength)
        guard frameCount > 0 else { return }

        // vDSP_measqv → mean(x²); sqrt(mean square) → RMS.
        var meanSquare: Float = 0.0
        vDSP_measqv(channelData, 1, &meanSquare, vDSP_Length(frameCount))
        currentRMS = sqrt(meanSquare)

        bufferLock.lock()
        pcmBuffer.append(contentsOf: UnsafeBufferPointer(start: channelData, count: frameCount))
        bufferLock.unlock()
    }

    // MARK: - stopRecording

    private func stopRecording(result: @escaping FlutterResult) {
        guard isRecording else {
            result(FlutterError(code: "NOT_RECORDING",
                                message: "No recording is in progress.", details: nil))
            return
        }

        isRecording = false
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine?.reset()
        engine = nil
        converter = nil

        // Non-fatal — lets other audio (music, calls) resume.
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)

        bufferLock.lock()
        let samples = pcmBuffer
        pcmBuffer.removeAll(keepingCapacity: false)
        bufferLock.unlock()
        currentRMS = 0.0

        // Reinterpret [Float] as raw bytes.
        // Flutter's standard codec decodes float32 typed data as Dart Float32List.
        let data = samples.withUnsafeBufferPointer { Data(buffer: $0) }
        result(FlutterStandardTypedData(float32: data))
    }

    private func extractFeatures(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let extractor = featureExtractor else {
            result(FlutterError(code: "EXTRACTOR_UNAVAILABLE",
                                message: "Feature extractor could not be initialized.",
                                details: nil))
            return
        }

        guard let args = call.arguments as? [String: Any] else {
            result(FlutterError(code: "BAD_ARGS",
                                message: "extractFeatures expects a map of arguments.",
                                details: nil))
            return
        }

        let sampleRate = (args["sampleRate"] as? NSNumber)?.floatValue ?? Float(targetSampleRate)

        guard sampleRate > 0 else {
            result(FlutterError(code: "BAD_SAMPLE_RATE",
                                message: "sampleRate must be greater than zero.",
                                details: nil))
            return
        }

        let signal: [Float]
        if let typed = args["pcm"] as? FlutterStandardTypedData {
            signal = typed.data.withUnsafeBytes { rawBuffer in
                let floatBuffer = rawBuffer.bindMemory(to: Float32.self)
                return Array(floatBuffer)
            }
        } else if let values = args["pcm"] as? [NSNumber] {
            signal = values.map { $0.floatValue }
        } else {
            result(FlutterError(code: "BAD_PCM",
                                message: "pcm must be Float32 typed data or a numeric list.",
                                details: nil))
            return
        }

        guard !signal.isEmpty else {
            result(FlutterError(code: "EMPTY_PCM",
                                message: "pcm must contain at least one sample.",
                                details: nil))
            return
        }

        featureExtractionQueue.async {
            let features = extractor.extract(signal: signal, sampleRate: sampleRate)
            let payload: [String: Any] = [
                "features": features.values.mapValues { Double($0) },
                "trace": features.trace
            ]
            DispatchQueue.main.async {
                result(payload)
            }
        }
    }

    private func extractFeaturesFromWavBytes(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let extractor = featureExtractor else {
            result(FlutterError(code: "EXTRACTOR_UNAVAILABLE",
                                message: "Feature extractor could not be initialized.",
                                details: nil))
            return
        }

        guard let args = call.arguments as? [String: Any],
              let typed = args["wavBytes"] as? FlutterStandardTypedData else {
            result(FlutterError(code: "BAD_ARGS",
                                message: "extractFeaturesFromWavBytes expects wavBytes typed data.",
                                details: nil))
            return
        }

        let wavData = typed.data
        guard !wavData.isEmpty else {
            result(FlutterError(code: "EMPTY_WAV",
                                message: "wavBytes must not be empty.",
                                details: nil))
            return
        }

        featureExtractionQueue.async {
            do {
                let payload = try self.loadResampledSignal(fromWavData: wavData)
                let features = extractor.extract(signal: payload.samples, sampleRate: payload.sampleRate)
                let response: [String: Any] = [
                    "features": features.values.mapValues { Double($0) },
                    "trace": features.trace
                ]
                DispatchQueue.main.async {
                    result(response)
                }
            } catch {
                DispatchQueue.main.async {
                    result(FlutterError(code: "WAV_EXTRACT_ERROR",
                                        message: "Could not decode WAV bytes: \(error.localizedDescription)",
                                        details: nil))
                }
            }
        }
    }

    private func loadResampledSignal(fromWavData wavData: Data) throws -> (samples: [Float], sampleRate: Float) {
        let tempURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        try wavData.write(to: tempURL, options: .atomic)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let file = try AVAudioFile(forReading: tempURL)
        let inputFormat = file.processingFormat
        guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                               sampleRate: targetSampleRate,
                                               channels: 1,
                                               interleaved: false),
              let conv = AVAudioConverter(from: inputFormat, to: targetFormat),
              let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat,
                                                 frameCapacity: AVAudioFrameCount(file.length)) else {
            throw NSError(domain: "AudioBridge", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not initialize WAV conversion"]) }

        try file.read(into: inputBuffer)
        let ratio = targetFormat.sampleRate / inputFormat.sampleRate
        let outCapacity = AVAudioFrameCount(ceil(Double(inputBuffer.frameLength) * ratio) + 1)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat,
                                                  frameCapacity: outCapacity) else {
            throw NSError(domain: "AudioBridge", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not create output buffer"]) }

        var consumed = false
        var convError: NSError?
        let status = conv.convert(to: outputBuffer, error: &convError) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if status == .error {
            throw convError ?? NSError(domain: "AudioBridge", code: 3, userInfo: [NSLocalizedDescriptionKey: "WAV conversion failed"]) }
        guard let channelData = outputBuffer.floatChannelData?[0] else {
            throw NSError(domain: "AudioBridge", code: 4, userInfo: [NSLocalizedDescriptionKey: "Missing converted channel data"]) }

        let samples = Array(UnsafeBufferPointer(start: channelData, count: Int(outputBuffer.frameLength)))
        return (samples, Float(targetFormat.sampleRate))
    }

    private func playAudioFile(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let path = args["path"] as? String else {
            result(FlutterError(code: "BAD_ARGS",
                                message: "playAudioFile expects a path argument.",
                                details: nil))
            return
        }

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default, options: [.duckOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)

            if currentAudioPath != path || audioPlayer == nil {
                let url = URL(fileURLWithPath: path)
                audioPlayer = try AVAudioPlayer(contentsOf: url)
                currentAudioPath = path
                audioPlayer?.prepareToPlay()
            }
            audioPlayer?.play()
            result(nil)
        } catch {
            result(FlutterError(code: "PLAYBACK_ERROR",
                                message: "Could not play saved audio: \(error.localizedDescription)",
                                details: nil))
        }
    }

    private func pauseAudioPlayback(result: @escaping FlutterResult) {
        audioPlayer?.pause()
        result(nil)
    }

    private func stopAudioPlayback(result: @escaping FlutterResult) {
        audioPlayer?.stop()
        audioPlayer?.currentTime = 0
        result(nil)
    }

    private func restartAudioPlayback(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let path = (call.arguments as? [String: Any])?["path"] as? String ?? currentAudioPath else {
            result(FlutterError(code: "BAD_ARGS",
                                message: "restartAudioPlayback expects a path or an existing player state.",
                                details: nil))
            return
        }

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default, options: [.duckOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)

            if currentAudioPath != path || audioPlayer == nil {
                let url = URL(fileURLWithPath: path)
                audioPlayer = try AVAudioPlayer(contentsOf: url)
                currentAudioPath = path
                audioPlayer?.prepareToPlay()
            }

            audioPlayer?.currentTime = 0
            audioPlayer?.play()
            result(nil)
        } catch {
            result(FlutterError(code: "PLAYBACK_ERROR",
                                message: "Could not restart saved audio: \(error.localizedDescription)",
                                details: nil))
        }
    }
}
