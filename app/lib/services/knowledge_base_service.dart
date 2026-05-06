import 'dart:convert';

import 'package:flutter/services.dart';

import '../models/knowledge_card.dart';

abstract final class KnowledgeBaseService {
  static const _screeningAsset = 'assets/knowledge/screening_reference.json';
  static const _doctorAsset = 'assets/knowledge/doctor_guide_reference.json';
  static const _educationTranslationsAsset =
      'assets/knowledge/education_reference_translations.json';

  static List<KnowledgeCard>? _screeningCards;
  static List<KnowledgeCard>? _doctorCards;
  static Map<String, Map<String, Map<String, String>>>? _educationTranslations;

  static Future<List<KnowledgeCard>> loadScreeningCards() async {
    if (_screeningCards != null) return _screeningCards!;
    _screeningCards = await _loadCards(_screeningAsset);
    return _screeningCards!;
  }

  static Future<List<KnowledgeCard>> loadDoctorCards() async {
    if (_doctorCards != null) return _doctorCards!;
    _doctorCards = await _loadCards(_doctorAsset);
    return _doctorCards!;
  }

  static Future<List<KnowledgeCard>> loadEducationCardsForRisk(
    String riskLabel, {
    String language = 'English',
  }) async {
    final cards = await loadScreeningCards();
    final ids = switch (riskLabel) {
      'low' => const [
        'screening_basics.overview',
        'screening_basics.why_voice_can_matter',
        'result_meaning.low',
        'result_meaning.low_next_step',
        'result_meaning.repeated_results',
        'limitations.not_diagnostic',
        'limitations.no_rule_out',
      ],
      'moderate' => const [
        'screening_basics.overview',
        'result_meaning.moderate',
        'result_meaning.moderate_when_to_follow_up',
        'confounders.common_overview',
        'limitations.full_model',
        'limitations.not_diagnostic',
      ],
      _ => const [
        'screening_basics.overview',
        'result_meaning.elevated',
        'result_meaning.elevated_balanced',
        'doctor_visit.when_to_discuss',
        'confounders.common_overview',
        'limitations.full_model',
        'limitations.not_diagnostic',
      ],
    };

    final selected = <KnowledgeCard>[];
    for (final id in ids) {
      final card = cards.where((c) => c.id == id).firstOrNull;
      if (card != null) selected.add(card);
    }
    return _localizeEducationCards(cards: selected, language: language);
  }

  static Future<void> reset() async {
    _screeningCards = null;
    _doctorCards = null;
    _educationTranslations = null;
    rootBundle.evict(_screeningAsset);
    rootBundle.evict(_doctorAsset);
    rootBundle.evict(_educationTranslationsAsset);
  }

  static Future<List<KnowledgeCard>> _loadCards(String assetPath) async {
    final raw = await rootBundle.loadString(assetPath);
    final json = jsonDecode(raw) as Map<String, dynamic>;
    final cards = json['cards'] as List<dynamic>? ?? const [];
    return cards
        .map((entry) => KnowledgeCard.fromJson(entry as Map<String, dynamic>))
        .toList(growable: false);
  }

  static Future<List<KnowledgeCard>> _localizeEducationCards({
    required List<KnowledgeCard> cards,
    required String language,
  }) async {
    if (language == 'English' || cards.isEmpty) return cards;

    final translations = await _loadEducationTranslations();
    final localized = translations[language];
    if (localized == null) return cards;

    return cards
        .map((card) {
          final translated = localized[card.id];
          if (translated == null) return card;
          return card.copyWith(
            title: translated['title'],
            body: translated['body'],
          );
        })
        .toList(growable: false);
  }

  static Future<Map<String, Map<String, Map<String, String>>>>
  _loadEducationTranslations() async {
    final cached = _educationTranslations;
    if (cached != null) return cached;

    final raw = await rootBundle.loadString(_educationTranslationsAsset);
    final json = jsonDecode(raw) as Map<String, dynamic>;
    final parsed = <String, Map<String, Map<String, String>>>{};

    for (final languageEntry in json.entries) {
      final cards =
          languageEntry.value as Map<String, dynamic>? ??
          const <String, dynamic>{};
      parsed[languageEntry.key] = cards.map((id, value) {
        final translation =
            value as Map<String, dynamic>? ?? const <String, dynamic>{};
        return MapEntry(id, {
          'title': translation['title'] as String? ?? '',
          'body': translation['body'] as String? ?? '',
        });
      });
    }

    _educationTranslations = parsed;
    return parsed;
  }
}
