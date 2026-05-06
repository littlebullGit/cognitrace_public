class DoctorDiscussionGuide {
  const DoctorDiscussionGuide({
    required this.visitReason,
    required this.resultSummary,
    required this.questionsToAsk,
    required this.contextToShare,
    required this.caveats,
  });

  final String visitReason;
  final String resultSummary;
  final List<String> questionsToAsk;
  final List<String> contextToShare;
  final List<String> caveats;

  factory DoctorDiscussionGuide.fromJson(Map<String, dynamic> json) {
    List<String> toList(Object? raw) {
      final list = raw as List<dynamic>? ?? const [];
      return list.map((value) => value.toString().trim()).toList();
    }

    return DoctorDiscussionGuide(
      visitReason: (json['visit_reason'] as String? ?? '').trim(),
      resultSummary: (json['result_summary'] as String? ?? '').trim(),
      questionsToAsk: toList(json['questions_to_ask']),
      contextToShare: toList(json['context_to_share']),
      caveats: toList(json['caveats']),
    );
  }

  bool get isComplete =>
      visitReason.isNotEmpty &&
      resultSummary.isNotEmpty &&
      questionsToAsk.isNotEmpty;

  String toShareText({
    required String title,
    required String visitReasonLabel,
    required String resultSummaryLabel,
    required String questionsLabel,
    required String contextLabel,
    required String caveatsLabel,
  }) {
    final buffer = StringBuffer()
      ..writeln(title)
      ..writeln();

    buffer.writeln('$visitReasonLabel\n$visitReason\n');
    buffer.writeln('$resultSummaryLabel\n$resultSummary\n');

    if (questionsToAsk.isNotEmpty) {
      buffer.writeln(questionsLabel);
      for (final question in questionsToAsk) {
        buffer.writeln('- $question');
      }
      buffer.writeln();
    }

    if (contextToShare.isNotEmpty) {
      buffer.writeln(contextLabel);
      for (final item in contextToShare) {
        buffer.writeln('- $item');
      }
      buffer.writeln();
    }

    if (caveats.isNotEmpty) {
      buffer.writeln(caveatsLabel);
      for (final caveat in caveats) {
        buffer.writeln('- $caveat');
      }
    }

    return buffer.toString().trimRight();
  }
}
