import 'package:cognitrace/models/doctor_discussion_guide.dart';
import 'package:cognitrace/services/gemma_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const sampleFeatures = <String, double>{
    'jitter_local': 0.032,
    'shimmer_local': 0.061,
    'f0_std': 18.4,
    'voiced_fraction': 0.78,
  };

  final sampleGuide = DoctorDiscussionGuide(
    visitReason:
        'I used a voice screening app and wanted to discuss the result.',
    resultSummary: 'Your screening showed a low risk score of 0.22.',
    questionsToAsk: [
      'Could you review my voice screening result?',
      'Should I repeat the screening later?',
    ],
    contextToShare: ['I recorded this at home in a quiet room.'],
    caveats: ['This is a screening tool, not a diagnosis.'],
  );

  // ---------------------------------------------------------------------------
  // Group 1: System prompt construction
  // ---------------------------------------------------------------------------

  group('practice system prompt', () {
    test('frames the session as practice or rehearsal, not a real consult', () {
      final prompt = GemmaService.debugPracticeSystemPrompt('English');

      expect(
        prompt.toLowerCase(),
        anyOf(contains('practice'), contains('rehearsal')),
      );
      expect(prompt.toLowerCase(), isNot(contains('you are a neurologist')));
    });

    test('includes a safety boundary stating the AI is not a doctor', () {
      final prompt = GemmaService.debugPracticeSystemPrompt('English');

      expect(
        prompt.toLowerCase(),
        anyOf(
          contains('not a doctor'),
          contains('not a real doctor'),
          contains('not a medical professional'),
        ),
      );
    });

    test('embeds the requested language directive', () {
      final prompt = GemmaService.debugPracticeSystemPrompt('Italiano');

      expect(prompt, contains('Italiano'));
    });

    test('instructs brief responses in the 2-3 sentence range', () {
      final prompt = GemmaService.debugPracticeSystemPrompt('English');

      expect(
        prompt,
        anyOf(
          contains('2-3 sentences'),
          contains('2 to 3 sentences'),
          contains('two to three sentences'),
        ),
      );
    });
  });

  // ---------------------------------------------------------------------------
  // Group 2: Practice prompt building
  // ---------------------------------------------------------------------------

  group('practice prompt building', () {
    test(
      'opening prompt includes SCREENING CONTEXT section and risk level',
      () {
        final prompt = GemmaService.debugBuildPracticePrompt(
          guide: sampleGuide,
          language: 'English',
          features: sampleFeatures,
          riskScore: 0.22,
          isOpening: true,
          isSummary: false,
        );

        expect(prompt, contains('SCREENING CONTEXT'));
        expect(prompt, anyOf(contains('0.22'), contains('low')));
      },
    );

    test(
      'opening prompt includes guide visit reason and first key question',
      () {
        final prompt = GemmaService.debugBuildPracticePrompt(
          guide: sampleGuide,
          language: 'English',
          features: sampleFeatures,
          riskScore: 0.22,
          isOpening: true,
          isSummary: false,
        );

        expect(prompt, contains(sampleGuide.visitReason));
        expect(prompt, contains(sampleGuide.questionsToAsk.first));
      },
    );

    test(
      'continuation prompt includes CONVERSATION SO FAR with history text',
      () {
        final history = [
          (role: 'user', text: 'Hello doctor.'),
          (role: 'model', text: 'Hello, how can I help you today?'),
        ];

        final prompt = GemmaService.debugBuildPracticePrompt(
          guide: sampleGuide,
          language: 'English',
          features: sampleFeatures,
          riskScore: 0.22,
          history: history,
          userMessage: 'I wanted to discuss my screening result.',
          isOpening: false,
          isSummary: false,
        );

        expect(prompt, contains('CONVERSATION SO FAR'));
        expect(prompt, contains('Hello doctor.'));
      },
    );

    test('history exceeding max window evicts early turns', () {
      // 5 exchanges = 10 turns, past the max window of 4 exchanges
      final longHistory = [
        (role: 'user', text: 'Turn 1 user.'),
        (role: 'model', text: 'Turn 1 model.'),
        (role: 'user', text: 'Turn 2 user.'),
        (role: 'model', text: 'Turn 2 model.'),
        (role: 'user', text: 'Turn 3 user.'),
        (role: 'model', text: 'Turn 3 model.'),
        (role: 'user', text: 'Turn 4 user.'),
        (role: 'model', text: 'Turn 4 model.'),
        (role: 'user', text: 'Turn 5 user.'),
        (role: 'model', text: 'Turn 5 model.'),
      ];

      final prompt = GemmaService.debugBuildPracticePrompt(
        guide: sampleGuide,
        language: 'English',
        features: sampleFeatures,
        riskScore: 0.22,
        history: longHistory,
        userMessage: 'One more follow-up question.',
        isOpening: false,
        isSummary: false,
      );

      // First exchange should be evicted from the window
      expect(prompt, isNot(contains('Turn 1 user.')));
    });

    test('prompt mentions EARLIER when history is truncated', () {
      final longHistory = [
        (role: 'user', text: 'Turn 1 user.'),
        (role: 'model', text: 'Turn 1 model.'),
        (role: 'user', text: 'Turn 2 user.'),
        (role: 'model', text: 'Turn 2 model.'),
        (role: 'user', text: 'Turn 3 user.'),
        (role: 'model', text: 'Turn 3 model.'),
        (role: 'user', text: 'Turn 4 user.'),
        (role: 'model', text: 'Turn 4 model.'),
        (role: 'user', text: 'Turn 5 user.'),
        (role: 'model', text: 'Turn 5 model.'),
      ];

      final prompt = GemmaService.debugBuildPracticePrompt(
        guide: sampleGuide,
        language: 'English',
        features: sampleFeatures,
        riskScore: 0.22,
        history: longHistory,
        userMessage: 'One more follow-up question.',
        isOpening: false,
        isSummary: false,
      );

      expect(prompt.toLowerCase(), contains('earlier'));
    });

    test('summary prompt lists covered and uncovered topics', () {
      final prompt = GemmaService.debugBuildPracticePrompt(
        guide: sampleGuide,
        language: 'English',
        features: sampleFeatures,
        riskScore: 0.22,
        isOpening: false,
        isSummary: true,
        topicsCovered: {RehearsalTopic.visitReason},
      );

      expect(
        prompt.toLowerCase(),
        anyOf(contains('covered'), contains('discussed')),
      );
      expect(
        prompt.toLowerCase(),
        anyOf(
          contains('not yet'),
          contains('uncovered'),
          contains('remaining'),
        ),
      );
    });
  });

  // ---------------------------------------------------------------------------
  // Group 3: Topic detection
  // ---------------------------------------------------------------------------

  group('topic detection', () {
    test('detects visitReason from screening / app / voice keywords', () {
      const triggers = [
        'I used an app to screen myself',
        'the voice app flagged something',
        'this screening result surprised me',
      ];

      for (final msg in triggers) {
        final topics = GemmaService.debugDetectPracticeTopics(msg, sampleGuide);
        expect(
          topics,
          contains(RehearsalTopic.visitReason),
          reason: 'expected visitReason for: $msg',
        );
      }
    });

    test('detects screeningResult from result / risk / score keywords', () {
      const triggers = [
        'what does my result mean?',
        'my risk level worries me a little',
        'the score was 0.22',
      ];

      for (final msg in triggers) {
        final topics = GemmaService.debugDetectPracticeTopics(msg, sampleGuide);
        expect(
          topics,
          contains(RehearsalTopic.screeningResult),
          reason: 'expected screeningResult for: $msg',
        );
      }
    });

    test('detects nextSteps from follow-up / repeat / refer keywords', () {
      const triggers = [
        'should I follow-up in six months?',
        'do I need to repeat the test?',
        'would you refer me to a specialist?',
      ];

      for (final msg in triggers) {
        final topics = GemmaService.debugDetectPracticeTopics(msg, sampleGuide);
        expect(
          topics,
          contains(RehearsalTopic.nextSteps),
          reason: 'expected nextSteps for: $msg',
        );
      }
    });

    test('returns empty set for a clearly unrelated message', () {
      final topics = GemmaService.debugDetectPracticeTopics(
        'the weather is nice',
        sampleGuide,
      );

      expect(topics, isEmpty);
    });
  });

  // ---------------------------------------------------------------------------
  // Group 4: Suggested responses
  // ---------------------------------------------------------------------------

  group('suggested practice responses', () {
    test('returns at least 2 responses when no topics covered', () {
      final responses = GemmaService.debugSuggestedPracticeResponses(
        guide: sampleGuide,
        language: 'English',
        topicsCovered: {},
      );

      expect(responses.length, greaterThanOrEqualTo(2));
    });

    test('returns at most 4 responses', () {
      final responses = GemmaService.debugSuggestedPracticeResponses(
        guide: sampleGuide,
        language: 'English',
        topicsCovered: {},
      );

      expect(responses.length, lessThanOrEqualTo(4));
    });

    test('always includes an option to end the practice session', () {
      final responses = GemmaService.debugSuggestedPracticeResponses(
        guide: sampleGuide,
        language: 'English',
        topicsCovered: {},
      );

      final hasEndOption = responses.any(
        (r) =>
            r.toLowerCase().contains('end') ||
            r.toLowerCase().contains('finish') ||
            r.toLowerCase().contains('stop'),
      );

      expect(
        hasEndOption,
        isTrue,
        reason: 'at least one suggestion should offer a way to end the session',
      );
    });

    test('adapts suggestions after topics are covered', () {
      final initial = GemmaService.debugSuggestedPracticeResponses(
        guide: sampleGuide,
        language: 'English',
        topicsCovered: {},
      );

      final afterCovered = GemmaService.debugSuggestedPracticeResponses(
        guide: sampleGuide,
        language: 'English',
        topicsCovered: {
          RehearsalTopic.visitReason,
          RehearsalTopic.screeningResult,
        },
      );

      expect(
        initial,
        isNot(equals(afterCovered)),
        reason: 'suggestions should change once key topics have been covered',
      );
    });
  });
}
