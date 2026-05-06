import 'package:cognitrace/services/gemma_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const sampleFeatures = <String, double>{
    'jitter_local': 0.032,
    'jitter_rap': 0.019,
    'jitter_ppq5': 0.021,
    'shimmer_local': 0.061,
    'shimmer_apq3': 0.044,
    'f0_std': 18.4,
    'voiced_fraction': 0.78,
    'rms_energy': 0.042,
  };

  group('doctor guide knowledge cards', () {
    test('uses the real low-risk doctor guide asset ids', () async {
      final ids = await GemmaService.debugDoctorGuideKnowledgeCardIds(
        features: sampleFeatures,
        riskScore: 0.22,
        language: 'English',
        userNotes:
            'noticed_changes: Voice feels softer\n'
            'recording_factors: Tired, noisy room',
      );

      expect(ids, contains('doctor_visit.opening_statement'));
      expect(ids, contains('doctor_visit.low_guidance'));
      expect(ids, contains('doctor_visit.context_to_share'));
      expect(ids, contains('doctor_visit.questions_about_context'));
      expect(ids, isNot(contains('doctor_visit.elevated_guidance')));
    });

    test('uses elevated guidance cards for elevated risk', () async {
      final ids = await GemmaService.debugDoctorGuideKnowledgeCardIds(
        features: sampleFeatures,
        riskScore: 0.84,
        language: 'English',
      );

      expect(ids, contains('doctor_visit.elevated_guidance'));
      expect(ids, contains('doctor_visit.repeat_screening_question'));
      expect(ids, contains('doctor_visit.follow_up_goal'));
      expect(ids, isNot(contains('doctor_visit.low_guidance')));
    });
  });

  group('fallback doctor discussion guide', () {
    test('threads structured user context into the guide', () {
      final guide = GemmaService.debugFallbackDoctorDiscussionGuide(
        riskLevel: 'moderate',
        riskScore: 0.56,
        language: 'English',
        userNotes:
            'visit_reason: My spouse noticed my voice is softer\n'
            'noticed_changes: Speaking takes more effort lately\n'
            'recording_factors: Tired, noisy room',
      );

      expect(guide.visitReason, 'My spouse noticed my voice is softer');
      expect(
        guide.contextToShare,
        contains('Changes I noticed: Speaking takes more effort lately'),
      );
      expect(
        guide.contextToShare,
        contains('Recording conditions to mention: Tired, noisy room'),
      );
      expect(guide.questionsToAsk.first, contains('moderate screening result'));
    });

    test('keeps low-risk fallback baseline-oriented', () {
      final guide = GemmaService.debugFallbackDoctorDiscussionGuide(
        riskLevel: 'low',
        riskScore: 0.18,
        language: 'English',
      );

      expect(guide.resultSummary, contains('baseline'));
      expect(guide.questionsToAsk.first, contains('voice or speech'));
    });

    test('explains elevated fallback with 56 user-facing voice markers', () {
      final guide = GemmaService.debugFallbackDoctorDiscussionGuide(
        riskLevel: 'elevated',
        riskScore: 0.84,
        language: 'English',
      );

      expect(guide.resultSummary, contains('56 voice markers'));
      expect(guide.resultSummary, isNot(contains('ensemble')));
      expect(guide.resultSummary, isNot(contains('3 model')));
    });
  });
}
