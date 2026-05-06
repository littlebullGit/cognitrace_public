class VoiceCaptureQualityResult {
  const VoiceCaptureQualityResult({
    required this.hasEnoughVoice,
    required this.progress,
  });

  final bool hasEnoughVoice;
  final double progress;
}

abstract final class VoiceCaptureQuality {
  static const clearPeakThreshold = 0.2;

  static VoiceCaptureQualityResult evaluate({
    required Duration detectedVoiceDuration,
    required Duration requiredVoiceDuration,
    required double peakDetectedLevel,
  }) {
    if (requiredVoiceDuration <= Duration.zero) {
      return const VoiceCaptureQualityResult(hasEnoughVoice: true, progress: 1);
    }

    final durationProgress =
        detectedVoiceDuration.inMicroseconds /
        requiredVoiceDuration.inMicroseconds;
    final hasClearPeak = peakDetectedLevel >= clearPeakThreshold;
    final hasEnoughVoice =
        detectedVoiceDuration >= requiredVoiceDuration || hasClearPeak;

    return VoiceCaptureQualityResult(
      hasEnoughVoice: hasEnoughVoice,
      progress: hasClearPeak ? 1 : durationProgress.clamp(0.0, 1.0),
    );
  }
}
