import 'package:cognitrace/models/doctor_discussion_guide.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('DoctorDiscussionGuide', () {
    test('fromJson parses the structured guide', () {
      final guide = DoctorDiscussionGuide.fromJson({
        'visit_reason': 'I received a screening result I want to discuss.',
        'result_summary': 'The app found patterns worth discussing.',
        'questions_to_ask': [
          'Should I repeat screening?',
          'Is follow-up needed?',
        ],
        'context_to_share': ['Recorded at home'],
        'caveats': ['Not a diagnosis'],
      });

      expect(
        guide.visitReason,
        'I received a screening result I want to discuss.',
      );
      expect(guide.resultSummary, 'The app found patterns worth discussing.');
      expect(guide.questionsToAsk, hasLength(2));
      expect(guide.contextToShare.single, 'Recorded at home');
      expect(guide.caveats.single, 'Not a diagnosis');
      expect(guide.isComplete, isTrue);
    });

    test('toShareText renders all sections', () {
      const guide = DoctorDiscussionGuide(
        visitReason: 'I want to discuss this screening result.',
        resultSummary: 'This was a moderate screening result.',
        questionsToAsk: ['Should I follow up?'],
        contextToShare: ['Recorded at home'],
        caveats: ['Not a diagnosis'],
      );

      final text = guide.toShareText(
        title: 'Doctor discussion guide',
        visitReasonLabel: 'Why I’m bringing this up',
        resultSummaryLabel: 'What the screening showed',
        questionsLabel: 'Questions I want to ask',
        contextLabel: 'Context that may matter',
        caveatsLabel: 'Important limits of the app',
      );

      expect(text, contains('Doctor discussion guide'));
      expect(text, contains('Why I’m bringing this up'));
      expect(text, contains('- Should I follow up?'));
      expect(text, contains('Important limits of the app'));
    });
  });
}
