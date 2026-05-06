import 'dart:async';

import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';

import '../l10n/app_strings.dart';
import '../models/doctor_discussion_guide.dart';
import '../services/gemma_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import 'doctor_practice_sheet.dart';

class DoctorGuideSheet extends StatefulWidget {
  const DoctorGuideSheet({super.key, required this.language});

  final String language;

  @override
  State<DoctorGuideSheet> createState() => _DoctorGuideSheetState();
}

class _DoctorGuideSheetState extends State<DoctorGuideSheet> {
  late final TextEditingController _visitReasonController;
  late final TextEditingController _changesController;
  final Set<String> _recordingFactors = <String>{};
  DoctorDiscussionGuide? _guide;
  String? _error;
  bool _loading = false;
  bool _sharing = false;

  String _s(String key) => AppStrings.get(key, widget.language);

  @override
  void initState() {
    super.initState();
    _visitReasonController = TextEditingController();
    _changesController = TextEditingController();
  }

  @override
  void dispose() {
    _visitReasonController.dispose();
    _changesController.dispose();
    super.dispose();
  }

  Future<void> _generateGuide() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final guide = await GemmaService.generateDoctorDiscussionGuide(
        language: widget.language,
        userNotes: _buildUserNotes(),
      );
      if (!mounted) return;
      setState(() {
        _guide = guide;
        _loading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _error = error.toString();
        _loading = false;
      });
    }
  }

  String? _buildUserNotes() {
    final lines = <String>[];
    final visitReason = _visitReasonController.text.trim();
    final noticedChanges = _changesController.text.trim();
    if (visitReason.isNotEmpty) {
      lines.add('visit_reason: $visitReason');
    }
    if (noticedChanges.isNotEmpty) {
      lines.add('noticed_changes: $noticedChanges');
    }
    if (_recordingFactors.isNotEmpty) {
      final factors = _recordingFactors.map(_recordingFactorLabel).join(', ');
      lines.add('recording_factors: $factors');
    }
    if (lines.isEmpty) return null;
    return lines.join('\n');
  }

  String _recordingFactorLabel(String factor) {
    return switch (factor) {
      'tired' => _s('doctor_factor_tired'),
      'sick' => _s('doctor_factor_sick'),
      'noisy_room' => _s('doctor_factor_noisy_room'),
      _ => factor,
    };
  }

  void _toggleFactor(String factor) {
    setState(() {
      if (_recordingFactors.contains(factor)) {
        _recordingFactors.remove(factor);
      } else {
        _recordingFactors.add(factor);
      }
    });
  }

  Widget _buildContextForm(ScrollController scrollController) {
    return ListView(
      controller: scrollController,
      padding: EdgeInsets.fromLTRB(
        20,
        18,
        20,
        MediaQuery.of(context).viewInsets.bottom + 24,
      ),
      children: [
        Text(
          _s('doctor_guide_personalize_body'),
          style: AppTextStyles.bodyMedium.copyWith(
            color: AppColors.textSecondary,
          ),
        ),
        const SizedBox(height: 20),
        Text(_s('doctor_guide_reason_label'), style: AppTextStyles.labelLarge),
        const SizedBox(height: 8),
        TextField(
          controller: _visitReasonController,
          decoration: InputDecoration(
            hintText: _s('doctor_guide_reason_hint'),
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
            alignLabelWithHint: true,
          ),
          minLines: 2,
          maxLines: 3,
          textCapitalization: TextCapitalization.sentences,
        ),
        const SizedBox(height: 16),
        Text(_s('doctor_guide_changes_label'), style: AppTextStyles.labelLarge),
        const SizedBox(height: 8),
        TextField(
          controller: _changesController,
          decoration: InputDecoration(
            hintText: _s('doctor_guide_changes_hint'),
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
            alignLabelWithHint: true,
          ),
          minLines: 2,
          maxLines: 3,
          textCapitalization: TextCapitalization.sentences,
        ),
        const SizedBox(height: 16),
        Text(
          _s('doctor_guide_recording_factors'),
          style: AppTextStyles.labelLarge,
        ),
        const SizedBox(height: 10),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _FactorChip(
              label: _s('doctor_factor_tired'),
              selected: _recordingFactors.contains('tired'),
              onSelected: () => _toggleFactor('tired'),
            ),
            _FactorChip(
              label: _s('doctor_factor_sick'),
              selected: _recordingFactors.contains('sick'),
              onSelected: () => _toggleFactor('sick'),
            ),
            _FactorChip(
              label: _s('doctor_factor_noisy_room'),
              selected: _recordingFactors.contains('noisy_room'),
              onSelected: () => _toggleFactor('noisy_room'),
            ),
          ],
        ),
        if (_error != null) ...[
          const SizedBox(height: 16),
          Text(
            _error!,
            style: AppTextStyles.bodyMedium.copyWith(
              color: AppColors.riskElevated,
            ),
          ),
        ],
        const SizedBox(height: 24),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton(
            onPressed: _loading ? null : _generateGuide,
            child: _loading
                ? Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                      const SizedBox(width: 10),
                      Flexible(child: Text(_s('generating_doctor_guide'))),
                    ],
                  )
                : Text(_s('generate_guide')),
          ),
        ),
      ],
    );
  }

  Widget _buildGuide(ScrollController scrollController) {
    return ListView(
      controller: scrollController,
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 24),
      children: [
        _Section(title: _s('visit_reason'), body: _guide!.visitReason),
        _Section(title: _s('result_summary'), body: _guide!.resultSummary),
        _ListSection(
          title: _s('questions_to_ask'),
          items: _guide!.questionsToAsk,
        ),
        _ListSection(
          title: _s('context_to_share'),
          items: _guide!.contextToShare,
        ),
        _ListSection(title: _s('caveats'), items: _guide!.caveats),
      ],
    );
  }

  void _startPractice(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) =>
          DoctorPracticeSheet(guide: _guide!, language: widget.language),
    );
  }

  Future<void> _shareGuide() async {
    if (_guide == null || _sharing) return;
    setState(() => _sharing = true);
    try {
      await Share.share(
        _guide!.toShareText(
          title: _s('doctor_discussion_guide'),
          visitReasonLabel: _s('visit_reason'),
          resultSummaryLabel: _s('result_summary'),
          questionsLabel: _s('questions_to_ask'),
          contextLabel: _s('context_to_share'),
          caveatsLabel: _s('caveats'),
        ),
      );
    } finally {
      if (mounted) setState(() => _sharing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      initialChildSize: 0.78,
      minChildSize: 0.45,
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
                      Icons.medical_information_outlined,
                      size: 20,
                      color: AppColors.primary,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _s('doctor_discussion_guide'),
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
                child: _guide == null
                    ? _buildContextForm(scrollController)
                    : _buildGuide(scrollController),
              ),
              if (!_loading && _error == null && _guide != null)
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 0, 20, 8),
                  child: SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: () => _startPractice(context),
                      icon: const Icon(
                        Icons.record_voice_over_outlined,
                        size: 18,
                      ),
                      label: Text(_s('practice_conversation')),
                    ),
                  ),
                ),
              if (!_loading && _error == null && _guide != null)
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
                  child: SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: _sharing ? null : _shareGuide,
                      icon: _sharing
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.share_outlined, size: 18),
                      label: Text(_s('share_guide')),
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }
}

class _FactorChip extends StatelessWidget {
  const _FactorChip({
    required this.label,
    required this.selected,
    required this.onSelected,
  });

  final String label;
  final bool selected;
  final VoidCallback onSelected;

  @override
  Widget build(BuildContext context) {
    return FilterChip(
      label: Text(label),
      selected: selected,
      onSelected: (_) => onSelected(),
      selectedColor: AppColors.primary.withValues(alpha: 0.14),
      checkmarkColor: AppColors.primary,
      labelStyle: AppTextStyles.bodyMedium.copyWith(
        color: selected ? AppColors.primary : AppColors.textPrimary,
      ),
      side: BorderSide(color: selected ? AppColors.primary : AppColors.border),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
    );
  }
}

class _Section extends StatelessWidget {
  const _Section({required this.title, required this.body});

  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: AppTextStyles.labelLarge),
          const SizedBox(height: 8),
          Text(body, style: AppTextStyles.bodyMedium),
        ],
      ),
    );
  }
}

class _ListSection extends StatelessWidget {
  const _ListSection({required this.title, required this.items});

  final String title;
  final List<String> items;

  @override
  Widget build(BuildContext context) {
    if (items.isEmpty) return const SizedBox.shrink();
    return Padding(
      padding: const EdgeInsets.only(bottom: 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: AppTextStyles.labelLarge),
          const SizedBox(height: 8),
          ...items.map(
            (item) => Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('\u2022 ', style: AppTextStyles.bodyMedium),
                  Expanded(child: Text(item, style: AppTextStyles.bodyMedium)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
