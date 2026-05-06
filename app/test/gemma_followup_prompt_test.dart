import 'package:cognitrace/services/gemma_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const features = <String, double>{
    'f0_mean': 145.0,
    'f0_std': 62.0,
    'voiced_fraction': 0.71,
    'jitter_local': 0.63,
    'jitter_rap': 0.58,
    'jitter_ppq5': 0.61,
    'shimmer_local': 3.42,
    'shimmer_apq3': 2.91,
    'hnr': 12.5,
    'rms_energy': 0.034,
    'log_energy': 8.42,
    'spec_centroid_mean': 1710.0,
    'spec_bandwidth_mean': 1380.0,
    'mfcc_0_mean': -520.0,
  };

  group('Gemma follow-up prompt builder', () {
    test(
      'feature question pulls focused feature context and card only',
      () async {
        final prompt = await GemmaService.debugBuildFollowUpPrompt(
          features: features,
          riskScore: 0.82,
          language: 'English',
          question: 'What does jitter mean?',
        );

        expect(prompt, contains('QUESTION:\nWhat does jitter mean?'));
        expect(prompt, contains('jitter (jitter_local)'));
        expect(prompt, contains('PERSONALIZED INTERPRETATION HINTS:'));
        expect(prompt, contains('Your jitter is 63.00% in this screening'));
        expect(
          prompt,
          contains(
            'This single measure does not determine the classifier result on its own',
          ),
        );
        expect(prompt, contains('feature_reference.jitter'));
        expect(prompt, isNot(contains('mfcc_0_mean')));
        expect(prompt, isNot(contains('spec_bandwidth_mean')));
      },
    );

    test('next-steps question pulls elevated doctor guidance', () async {
      final cardIds = await GemmaService.debugKnowledgeCardIds(
        features: features,
        riskScore: 0.82,
        language: 'English',
        question: 'Should I see a doctor next?',
      );

      expect(cardIds, contains('result_meaning.elevated'));
      expect(cardIds, contains('doctor_visit.when_to_discuss'));
    });

    test(
      'prompt keeps compact state and notes all 56 values stay available',
      () async {
        final prompt = await GemmaService.debugBuildFollowUpPrompt(
          features: features,
          riskScore: 0.55,
          language: 'English',
          question: 'Why is my result moderate?',
        );

        expect(prompt, contains('CURRENT SCREENING STATE:'));
        expect(prompt, contains('screening basis: 56 acoustic voice markers'));
        expect(
          prompt,
          contains('marker table: 56 values available for drill-down'),
        );
        expect(prompt, isNot(contains('3-model ensemble')));
        expect(prompt, contains('REFERENCE CARDS:'));
        expect(prompt, contains('limitations.full_model'));
        expect(prompt, isNot(contains('VOICE STEADINESS:')));
      },
    );

    test('suggested questions adapt to risk and strongest feature', () {
      final suggestions = GemmaService.debugSuggestedQuestions(
        features: features,
        riskScore: 0.82,
        language: 'English',
      );

      expect(suggestions, contains('What does this elevated result mean?'));
      expect(
        suggestions,
        contains(
          'Could fatigue, noise, or dehydration have affected my result?',
        ),
      );
      expect(
        suggestions,
        contains('What can this app not tell me with certainty?'),
      );
      expect(
        suggestions,
        contains('How should I prepare to discuss this with a doctor?'),
      );
      expect(suggestions, contains('What does my jitter value mean for me?'));
    });

    test('contextual hnr does not become the fallback highlighted signal', () {
      const lowSignalFeatures = <String, double>{
        'f0_mean': 145.0,
        'f0_std': 0.08,
        'voiced_fraction': 0.92,
        'jitter_local': 0.0030,
        'jitter_rap': 0.0035,
        'jitter_ppq5': 0.0040,
        'shimmer_local': 0.0200,
        'shimmer_apq3': 0.0200,
        'hnr': -20.0,
        'rms_energy': 0.030,
        'log_energy': 8.4,
        'spec_centroid_mean': 1700.0,
        'spec_bandwidth_mean': 1380.0,
      };

      final suggestions = GemmaService.debugSuggestedQuestions(
        features: lowSignalFeatures,
        riskScore: 0.25,
        language: 'English',
      );

      expect(
        suggestions.any(
          (q) => q.toLowerCase().contains('harmonics-to-noise ratio'),
        ),
        isFalse,
      );
    });

    test('low-risk suggestions emphasize education and checking later', () {
      final suggestions = GemmaService.debugSuggestedQuestions(
        features: features,
        riskScore: 0.20,
        language: 'English',
      );

      expect(suggestions, contains('What in my result looked reassuring?'));
      expect(
        suggestions,
        contains('Why is voice used in this kind of screening?'),
      );
      expect(
        suggestions,
        contains('When would it make sense to check again later?'),
      );
      expect(
        suggestions.any(
          (q) => q.contains('prepare to discuss this with a doctor'),
        ),
        isFalse,
      );
    });
  });
}
