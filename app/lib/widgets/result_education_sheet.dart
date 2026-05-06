import 'package:flutter/material.dart';

import '../l10n/app_strings.dart';
import '../models/knowledge_card.dart';
import '../services/gemma_service.dart';
import '../services/knowledge_base_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

class ResultEducationSheet extends StatefulWidget {
  const ResultEducationSheet({
    super.key,
    required this.language,
    required this.riskLabel,
    required this.riskScore,
    required this.modelScores,
  });

  final String language;
  final String riskLabel;
  final double? riskScore;
  final Map<String, double>? modelScores;

  @override
  State<ResultEducationSheet> createState() => _ResultEducationSheetState();
}

class _ResultEducationSheetState extends State<ResultEducationSheet> {
  late Future<List<KnowledgeCard>> _cardsFuture;
  late Future<String?> _introFuture;

  String _s(String key) => AppStrings.get(key, widget.language);

  @override
  void initState() {
    super.initState();
    _cardsFuture = _loadCards();
    _introFuture = _loadIntro();
  }

  Future<List<KnowledgeCard>> _loadCards() async {
    return KnowledgeBaseService.loadEducationCardsForRisk(
      widget.riskLabel,
      language: widget.language,
    );
  }

  Future<String?> _loadIntro() async {
    final riskScore = widget.riskScore;
    if (riskScore == null) return null;
    return GemmaService.generateEducationIntro(
      riskScore: riskScore,
      language: widget.language,
      modelScores: widget.modelScores ?? const {},
    );
  }

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      initialChildSize: 0.78,
      minChildSize: 0.4,
      maxChildSize: 0.96,
      expand: false,
      builder: (context, scrollController) {
        return Container(
          decoration: const BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Column(
            children: [
              const SizedBox(height: 12),
              Center(
                child: Container(
                  width: 36,
                  height: 4,
                  decoration: BoxDecoration(
                    color: AppColors.border,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Row(
                  children: [
                    const Icon(
                      Icons.menu_book_outlined,
                      size: 20,
                      color: AppColors.primary,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _s('learn_about_result'),
                        style: AppTextStyles.headingSmall,
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.close_rounded, size: 22),
                      color: AppColors.textSecondary,
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ],
                ),
              ),
              const Divider(height: 1, color: AppColors.border),
              Expanded(
                child: FutureBuilder<List<KnowledgeCard>>(
                  future: _cardsFuture,
                  builder: (context, snapshot) {
                    if (snapshot.hasError) {
                      return Center(
                        child: Padding(
                          padding: const EdgeInsets.all(24),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                _s('education_unavailable'),
                                style: AppTextStyles.bodyMedium.copyWith(
                                  color: AppColors.riskElevated,
                                ),
                                textAlign: TextAlign.center,
                              ),
                              const SizedBox(height: 12),
                              OutlinedButton(
                                onPressed: () {
                                  setState(() {
                                    _cardsFuture = _loadCards();
                                  });
                                },
                                child: Text(_s('retry')),
                              ),
                            ],
                          ),
                        ),
                      );
                    }
                    if (!snapshot.hasData) {
                      return Center(
                        child: Padding(
                          padding: const EdgeInsets.all(24),
                          child: Text(
                            _s('loading_education'),
                            style: AppTextStyles.bodyMedium,
                          ),
                        ),
                      );
                    }
                    final cards = snapshot.data!;
                    if (cards.isEmpty) {
                      return Center(
                        child: Padding(
                          padding: const EdgeInsets.all(24),
                          child: Text(
                            _s('education_unavailable'),
                            style: AppTextStyles.bodyMedium,
                            textAlign: TextAlign.center,
                          ),
                        ),
                      );
                    }
                    return ListView(
                      controller: scrollController,
                      padding: const EdgeInsets.fromLTRB(20, 18, 20, 24),
                      children: [
                        FutureBuilder<String?>(
                          future: _introFuture,
                          builder: (context, introSnapshot) {
                            final intro = introSnapshot.data;
                            if (intro == null || intro.isEmpty) {
                              return const SizedBox.shrink();
                            }
                            return Container(
                              padding: const EdgeInsets.all(16),
                              margin: const EdgeInsets.only(bottom: 18),
                              decoration: BoxDecoration(
                                color: AppColors.primaryContainer,
                                borderRadius: BorderRadius.circular(12),
                                border: Border.all(
                                  color: AppColors.primary.withAlpha(32),
                                ),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    _s('for_you_right_now'),
                                    style: AppTextStyles.labelLarge.copyWith(
                                      color: AppColors.primary,
                                    ),
                                  ),
                                  const SizedBox(height: 8),
                                  Text(intro, style: AppTextStyles.bodyMedium),
                                ],
                              ),
                            );
                          },
                        ),
                        if (widget.riskLabel == 'low')
                          Container(
                            padding: const EdgeInsets.all(16),
                            margin: const EdgeInsets.only(bottom: 18),
                            decoration: BoxDecoration(
                              color: AppColors.riskLowContainer,
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(
                                color: AppColors.riskLow.withAlpha(40),
                              ),
                            ),
                            child: Text(
                              _s('baseline_saved_hint'),
                              style: AppTextStyles.bodyMedium,
                            ),
                          ),
                        ...cards.map(
                          (card) => Padding(
                            padding: const EdgeInsets.only(bottom: 18),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  card.title,
                                  style: AppTextStyles.labelLarge,
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  card.body,
                                  style: AppTextStyles.bodyMedium,
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
