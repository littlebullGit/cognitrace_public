import 'package:cognitrace/services/voice_metric_policy.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('VoiceMetricPolicy', () {
    test('formats jitter/shimmer ratios as percentages for display', () {
      expect(VoiceMetricPolicy.formatValue('jitter_local', 0.0869), '8.69%');
      expect(VoiceMetricPolicy.formatValue('shimmer_local', 0.1123), '11.23%');
    });

    test('compares raw ratios against raw reference thresholds', () {
      expect(
        VoiceMetricPolicy.isReferenceHigh('jitter_local', 0.0035),
        isFalse,
      );
      expect(VoiceMetricPolicy.isReferenceHigh('jitter_local', 0.0120), isTrue);
      expect(VoiceMetricPolicy.isReferenceHigh('jitter_rap', 0.0050), isFalse);
      expect(VoiceMetricPolicy.isReferenceHigh('jitter_rap', 0.0075), isTrue);
    });

    test('only classic perturbation subset keeps reference thresholds', () {
      expect(VoiceMetricPolicy.hasReferenceThreshold('jitter_local'), isTrue);
      expect(VoiceMetricPolicy.hasReferenceThreshold('jitter_rap'), isTrue);
      expect(VoiceMetricPolicy.hasReferenceThreshold('shimmer_local'), isFalse);
      expect(VoiceMetricPolicy.hasReferenceThreshold('shimmer_apq3'), isFalse);
      expect(VoiceMetricPolicy.hasReferenceThreshold('hnr'), isFalse);
    });

    test('formats hnr as contextual dB value only', () {
      expect(VoiceMetricPolicy.formatValue('hnr', -20.0), '-20.00 dB');
      expect(VoiceMetricPolicy.annotation('hnr', -20.0), isEmpty);
    });

    test('annotation text uses reference flag wording', () {
      expect(
        VoiceMetricPolicy.annotation('jitter_local', 0.0120),
        contains('ABOVE REF'),
      );
      expect(
        VoiceMetricPolicy.annotation('jitter_local', 0.0030),
        contains('WITHIN REF'),
      );
      expect(VoiceMetricPolicy.annotation('shimmer_local', 0.0400), isEmpty);
    });
  });
}
