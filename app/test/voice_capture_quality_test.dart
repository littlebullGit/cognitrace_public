import 'package:cognitrace/services/voice_capture_quality.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('VoiceCaptureQuality', () {
    test(
      'requires sustained detected voice before marking capture complete',
      () {
        const required = Duration(milliseconds: 1200);

        final quiet = VoiceCaptureQuality.evaluate(
          detectedVoiceDuration: const Duration(milliseconds: 400),
          requiredVoiceDuration: required,
          peakDetectedLevel: 0.14,
        );
        final enough = VoiceCaptureQuality.evaluate(
          detectedVoiceDuration: required,
          requiredVoiceDuration: required,
          peakDetectedLevel: 0.14,
        );

        expect(quiet.hasEnoughVoice, isFalse);
        expect(quiet.progress, closeTo(1 / 3, 0.001));
        expect(enough.hasEnoughVoice, isTrue);
        expect(enough.progress, 1.0);
      },
    );

    test('accepts a clear voice peak even before duration target', () {
      final result = VoiceCaptureQuality.evaluate(
        detectedVoiceDuration: const Duration(milliseconds: 80),
        requiredVoiceDuration: const Duration(milliseconds: 1200),
        peakDetectedLevel: 0.2,
      );

      expect(result.hasEnoughVoice, isTrue);
      expect(result.progress, 1.0);
    });
  });
}
