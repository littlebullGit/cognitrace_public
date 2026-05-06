import 'package:cognitrace/services/knowledge_base_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  tearDown(() async {
    await KnowledgeBaseService.reset();
  });

  group('KnowledgeBaseService education cards', () {
    test('returns reviewed English cards by default', () async {
      final cards = await KnowledgeBaseService.loadEducationCardsForRisk('low');
      final overview = cards.firstWhere(
        (card) => card.id == 'screening_basics.overview',
      );

      expect(overview.title, 'What this screening does');
      expect(overview.body, contains('voice screening tool'));
    });

    test(
      'returns prelocalized Italian cards without runtime translation',
      () async {
        final cards = await KnowledgeBaseService.loadEducationCardsForRisk(
          'low',
          language: 'Italiano',
        );
        final overview = cards.firstWhere(
          (card) => card.id == 'screening_basics.overview',
        );
        final lowMeaning = cards.firstWhere(
          (card) => card.id == 'result_meaning.low',
        );

        expect(overview.title, 'Cosa fa questo screening');
        expect(overview.body, contains('strumento di screening vocale'));
        expect(lowMeaning.title, 'Cosa significa un risultato basso');
      },
    );

    test('returns prelocalized Chinese cards for elevated education', () async {
      final cards = await KnowledgeBaseService.loadEducationCardsForRisk(
        'elevated',
        language: '中文',
      );
      final elevated = cards.firstWhere(
        (card) => card.id == 'result_meaning.elevated',
      );
      final clinician = cards.firstWhere(
        (card) => card.id == 'doctor_visit.when_to_discuss',
      );

      expect(elevated.title, '较高结果意味着什么');
      expect(elevated.body, contains('筛查信号'));
      expect(clinician.title, '什么时候适合和医生讨论');
    });
  });
}
