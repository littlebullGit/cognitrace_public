class KnowledgeCard {
  const KnowledgeCard({
    required this.id,
    required this.category,
    required this.audience,
    required this.title,
    required this.body,
    required this.tags,
    required this.riskLevels,
    required this.featureKeys,
    required this.safetyNotes,
  });

  final String id;
  final String category;
  final String audience;
  final String title;
  final String body;
  final List<String> tags;
  final List<String> riskLevels;
  final List<String> featureKeys;
  final List<String> safetyNotes;

  KnowledgeCard copyWith({String? title, String? body}) {
    return KnowledgeCard(
      id: id,
      category: category,
      audience: audience,
      title: title ?? this.title,
      body: body ?? this.body,
      tags: tags,
      riskLevels: riskLevels,
      featureKeys: featureKeys,
      safetyNotes: safetyNotes,
    );
  }

  factory KnowledgeCard.fromJson(Map<String, dynamic> json) {
    List<String> toStringList(Object? raw) {
      final list = raw as List<dynamic>? ?? const [];
      return list.map((value) => value.toString()).toList();
    }

    return KnowledgeCard(
      id: json['id'] as String,
      category: json['category'] as String,
      audience: json['audience'] as String? ?? 'patient',
      title: json['title'] as String,
      body: json['body'] as String,
      tags: toStringList(json['tags']),
      riskLevels: toStringList(json['risk_levels']),
      featureKeys: toStringList(json['feature_keys']),
      safetyNotes: toStringList(json['safety_notes']),
    );
  }
}
