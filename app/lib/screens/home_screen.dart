import 'dart:async';
import 'dart:typed_data';
import 'package:intl/intl.dart';

import 'package:flutter/material.dart';

import '../l10n/app_strings.dart';
import '../models/check_record.dart';
import '../navigation/app_router.dart';
import '../services/audio_archive_service.dart';
import '../services/audio_service.dart';
import '../services/check_history_service.dart';
import '../services/gemma_download_manager.dart';
import '../services/gemma_service.dart';
import '../services/language_preference_service.dart';
import '../services/sample_audio_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// Home screen — the returning-user entry point.
///
/// Contains:
///   • Hero card with "Start voice check" CTA
///   • Language selector (sheet picker)
///   • Previous checks list (placeholder data)
///   • Disclaimer footer
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _language = LanguagePreferenceService.defaultLanguage;
  late Future<List<CheckRecord>> _historyFuture;
  String? _activePlaybackId;

  @override
  void initState() {
    super.initState();
    _historyFuture = CheckHistoryService.load();
    unawaited(_loadLanguage());
    GemmaDownloadManager.instance.addListener(_onGemmaUpdate);
  }

  @override
  void dispose() {
    GemmaDownloadManager.instance.removeListener(_onGemmaUpdate);
    super.dispose();
  }

  void _onGemmaUpdate() {
    if (mounted) setState(() {});
  }

  Future<void> _loadLanguage() async {
    final lang = await LanguagePreferenceService.load();
    if (!mounted) return;
    setState(() => _language = lang);
  }

  String _s(String key) => AppStrings.get(key, _language);

  Future<void> _reloadHistory() async {
    setState(() {
      _historyFuture = CheckHistoryService.load();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('CogniTrace'),
        actions: [
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.settings_outlined),
            onPressed: () async {
              await Navigator.of(context).pushNamed(AppRoutes.settings);
              if (mounted) unawaited(_reloadHistory());
            },
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildHeroCard(context),
              const SizedBox(height: 16),
              _buildGemmaStatusCard(),
              const SizedBox(height: 24),
              _buildLanguageSelector(context),
              const SizedBox(height: 24),
              const Divider(),
              const SizedBox(height: 20),
              _buildHistory(),
              const SizedBox(height: 24),
              _buildDisclaimer(),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeroCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(_s('hero_tagline'), style: AppTextStyles.bodyLarge),
          const SizedBox(height: 24),
          ElevatedButton(
            onPressed: () => Navigator.of(context).pushNamed(AppRoutes.record),
            child: Text(_s('start_voice_check')),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Icon(
                Icons.person_outline,
                size: 16,
                color: AppColors.textSecondary,
              ),
              const SizedBox(width: 6),
              Text(
                _s('age_guidance'),
                style: AppTextStyles.caption.copyWith(
                  color: AppColors.textSecondary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          OutlinedButton(
            onPressed: () => _showSampleRunner(context),
            child: Text(_s('run_bundled_sample')),
          ),
        ],
      ),
    );
  }

  Widget _buildGemmaStatusCard() {
    final mgr = GemmaDownloadManager.instance;
    // Don't show card once model is ready.
    if (mgr.state == GemmaModelState.ready) {
      return const SizedBox.shrink();
    }

    final isDownloading = mgr.state == GemmaModelState.downloading;
    final isError = mgr.state == GemmaModelState.error;
    final pct = (mgr.progress * 100).toStringAsFixed(0);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isError
            ? AppColors.riskElevatedContainer
            : AppColors.primaryContainer,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: isError
              ? AppColors.riskElevated.withAlpha(48)
              : AppColors.primary.withAlpha(40),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                isError
                    ? Icons.error_outline_rounded
                    : Icons.auto_awesome_rounded,
                size: 18,
                color: isError ? AppColors.riskElevated : AppColors.primary,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  isError
                      ? _s('download_failed')
                      : isDownloading
                      ? _s('downloading_gemma').replaceAll('{pct}', pct)
                      : _s('download_gemma'),
                  style: AppTextStyles.bodyMedium.copyWith(
                    color: isError
                        ? AppColors.riskElevated
                        : AppColors.textPrimary,
                  ),
                ),
              ),
            ],
          ),
          if (isDownloading && mgr.progress > 0) ...[
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: mgr.progress,
                minHeight: 6,
                color: AppColors.primary,
                backgroundColor: AppColors.primary.withAlpha(30),
              ),
            ),
          ],
          if (isError) ...[
            const SizedBox(height: 10),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton(
                onPressed: () => unawaited(mgr.retry()),
                child: Text(_s('retry_download')),
              ),
            ),
          ],
        ],
      ),
    );
  }

  void _showSampleRunner(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => Padding(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
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
            const SizedBox(height: 20),
            Text(_s('run_bundled_title'), style: AppTextStyles.headingMedium),
            const SizedBox(height: 10),
            Text(
              _s('run_bundled_desc'),
              style: AppTextStyles.bodyMediumSecondary,
            ),
            const SizedBox(height: 16),
            ...SampleAudioService.clips.map((clip) {
              final title = clip.label == 'PD'
                  ? _s('ref_positive')
                  : _s('ref_control');
              return ListTile(
                contentPadding: EdgeInsets.zero,
                title: Text(title, style: AppTextStyles.bodyMedium),
                subtitle: Text(clip.label, style: AppTextStyles.caption),
                trailing: const Icon(
                  Icons.play_arrow_rounded,
                  color: AppColors.primary,
                ),
                onTap: () {
                  Navigator.of(context).pop();
                  unawaited(_runSampleClip(context, clip));
                },
              );
            }),
          ],
        ),
      ),
    );
  }

  Future<void> _runSampleClip(
    BuildContext context,
    SampleAudioClip clip,
  ) async {
    final messenger = ScaffoldMessenger.of(context);
    messenger.showSnackBar(SnackBar(content: Text(_s('loading_sample'))));
    try {
      final payload = await SampleAudioService.loadClip(clip);
      if (!context.mounted) return;
      Navigator.of(context).pushNamed(
        AppRoutes.analysis,
        arguments: AnalysisArguments(
          recordedPcm: payload.combinedPcm,
          sampleRate: payload.sampleRate,
          isReferenceSample: true,
          referenceWavBytes: payload.combinedWavBytes,
          taskPcmList: payload.taskPcmList,
          taskSampleLengths: payload.taskSampleLengths,
        ),
      );
    } catch (error) {
      messenger.showSnackBar(
        SnackBar(content: Text('Could not load sample: $error')),
      );
    }
  }

  Widget _buildLanguageSelector(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(
              Icons.language_rounded,
              size: 18,
              color: AppColors.primary,
            ),
            const SizedBox(width: 8),
            Text(
              '${_s('assessment_language')}:',
              style: AppTextStyles.labelLarge,
            ),
          ],
        ),
        const SizedBox(height: 10),
        GestureDetector(
          onTap: () => _showLanguagePicker(context),
          behavior: HitTestBehavior.opaque,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppColors.border),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Text(_language, style: AppTextStyles.bodyMedium),
                ),
                const Icon(
                  Icons.keyboard_arrow_down_rounded,
                  size: 20,
                  color: AppColors.textSecondary,
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  void _showLanguagePicker(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => Padding(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 40),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
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
            const SizedBox(height: 20),
            Text(_s('assessment_language'), style: AppTextStyles.headingMedium),
            const SizedBox(height: 12),
            ...LanguagePreferenceService.displayNames.map(
              (lang) => ListTile(
                contentPadding: EdgeInsets.zero,
                title: Text(lang, style: AppTextStyles.bodyMedium),
                trailing: _language == lang
                    ? const Icon(Icons.check_rounded, color: AppColors.primary)
                    : null,
                onTap: () {
                  setState(() => _language = lang);
                  unawaited(LanguagePreferenceService.save(lang));
                  Navigator.pop(context);
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHistory() {
    return FutureBuilder<List<CheckRecord>>(
      future: _historyFuture,
      builder: (context, snapshot) {
        final history = snapshot.data ?? const <CheckRecord>[];
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(_s('previous_checks'), style: AppTextStyles.headingSmall),
            const SizedBox(height: 12),
            if (history.isEmpty)
              Text(
                _s('no_previous_checks'),
                style: AppTextStyles.bodyMediumSecondary,
              )
            else
              Container(
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppColors.border),
                ),
                child: Column(
                  children: List.generate(history.length, (i) {
                    final entry = history[i];
                    return Column(
                      children: [
                        Dismissible(
                          key: ValueKey(entry.id),
                          direction: DismissDirection.endToStart,
                          background: Container(
                            alignment: Alignment.centerRight,
                            padding: const EdgeInsets.symmetric(horizontal: 20),
                            color: AppColors.riskElevatedContainer,
                            child: const Icon(
                              Icons.delete_outline,
                              color: AppColors.riskElevated,
                            ),
                          ),
                          onDismissed: (_) async {
                            await AudioArchiveService.deleteAudio(
                              entry.audioFilePath,
                            );
                            await CheckHistoryService.delete(entry.id);
                            await _reloadHistory();
                          },
                          child: _HistoryRow(
                            entry: entry,
                            isPlaying: _activePlaybackId == entry.id,
                            language: _language,
                            onPlay: () =>
                                unawaited(_playSavedRun(context, entry)),
                            onPause: () =>
                                unawaited(_pauseSavedRun(context, entry)),
                            onRerun: () =>
                                unawaited(_rerunSavedRun(context, entry)),
                            onEditNotes: () => _editNotes(context, entry),
                            onTap: () {
                              Navigator.of(context).pushNamed(
                                AppRoutes.results,
                                arguments: entry.toResultsArguments(),
                              );
                            },
                          ),
                        ),
                        if (i < history.length - 1)
                          const Divider(height: 1, indent: 16, endIndent: 16),
                      ],
                    );
                  }),
                ),
              ),
          ],
        );
      },
    );
  }

  Future<void> _playSavedRun(BuildContext context, CheckRecord entry) async {
    final messenger = ScaffoldMessenger.of(context);
    if (!await AudioArchiveService.hasAudio(entry.audioFilePath)) {
      messenger.showSnackBar(
        const SnackBar(
          content: Text(
            'This older saved result does not include local audio. Run a new check to enable playback.',
          ),
        ),
      );
      return;
    }
    try {
      final resolvedPath = await AudioArchiveService.resolvePath(
        entry.audioFilePath,
      );
      await playAudioFile(resolvedPath);
      if (!mounted) return;
      setState(() {
        _activePlaybackId = entry.id;
      });
    } catch (error) {
      messenger.showSnackBar(
        SnackBar(content: Text('Could not play saved audio: $error')),
      );
    }
  }

  Future<void> _pauseSavedRun(BuildContext context, CheckRecord entry) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      await pauseAudioPlayback();
      if (!mounted) return;
      setState(() {
        if (_activePlaybackId == entry.id) {
          _activePlaybackId = null;
        }
      });
    } catch (error) {
      messenger.showSnackBar(
        SnackBar(content: Text('Could not pause saved audio: $error')),
      );
    }
  }

  Future<void> _rerunSavedRun(BuildContext context, CheckRecord entry) async {
    final messenger = ScaffoldMessenger.of(context);
    if (!await AudioArchiveService.hasAudio(entry.audioFilePath)) {
      messenger.showSnackBar(
        const SnackBar(
          content: Text(
            'This older saved result does not include local audio. Run a new check to enable rerun.',
          ),
        ),
      );
      return;
    }
    try {
      final wavBytes = await AudioArchiveService.readWavBytes(
        entry.audioFilePath,
      );

      // Reconstruct per-task PCM from saved task boundaries.
      List<Float32List>? taskPcmList;
      Float32List recordedPcm = Float32List(0);
      final lengths = entry.taskSampleLengths;
      if (lengths != null && lengths.isNotEmpty) {
        final allPcm = AudioArchiveService.decodePcmFromWav(wavBytes);
        recordedPcm = allPcm;
        taskPcmList = <Float32List>[];
        var offset = 0;
        for (final len in lengths) {
          final end = (offset + len).clamp(0, allPcm.length);
          taskPcmList.add(Float32List.sublistView(allPcm, offset, end));
          offset = end;
        }
      }

      if (!context.mounted) return;
      Navigator.of(context).pushNamed(
        AppRoutes.analysis,
        arguments: AnalysisArguments(
          recordedPcm: recordedPcm,
          sampleRate: 16000,
          isReferenceSample: lengths == null,
          referenceWavBytes: lengths == null ? wavBytes : null,
          sourceTag: 'saved',
          shouldAutoSave: false,
          taskPcmList: taskPcmList,
          taskSampleLengths: lengths,
        ),
      );
    } catch (error) {
      messenger.showSnackBar(
        SnackBar(content: Text('Could not reopen saved audio: $error')),
      );
    }
  }

  void _editNotes(BuildContext context, CheckRecord record) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _NotesEditor(
        record: record,
        language: _language,
        onSave: (updated) async {
          await CheckHistoryService.save(updated);
          if (mounted) setState(() {});
        },
      ),
    );
  }

  Widget _buildDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surfaceElevated,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.info_outline_rounded,
            size: 18,
            color: AppColors.textSecondary,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(_s('disclaimer_long'), style: AppTextStyles.caption),
          ),
        ],
      ),
    );
  }
}

// ── Data model ────────────────────────────────────────────────────────────────

class _HistoryRow extends StatelessWidget {
  final CheckRecord entry;
  final bool isPlaying;
  final String language;
  final VoidCallback onTap;
  final VoidCallback onPlay;
  final VoidCallback onPause;
  final VoidCallback onRerun;
  final VoidCallback onEditNotes;

  const _HistoryRow({
    required this.entry,
    required this.isPlaying,
    required this.language,
    required this.onTap,
    required this.onPlay,
    required this.onPause,
    required this.onRerun,
    required this.onEditNotes,
  });

  Color get _dotColor => switch (entry.riskLabel) {
    'low' => AppColors.riskLow,
    'moderate' => AppColors.riskModerate,
    _ => AppColors.riskElevated,
  };

  String get _label => switch (entry.riskLabel) {
    'low' => AppStrings.get('low_risk', language),
    _ => AppStrings.get('patterns_detected', language),
  };

  String get _sourceLabel => switch (entry.source) {
    'bundled_sample' => AppStrings.get('source_sample', language),
    'analysis' => AppStrings.get('source_live', language),
    _ => AppStrings.get('source_saved', language),
  };

  @override
  Widget build(BuildContext context) {
    final date = DateFormat('MMM d').format(entry.timestamp);
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text(date, style: AppTextStyles.bodyMedium),
                const SizedBox(width: 14),
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: _dotColor,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(_label, style: AppTextStyles.bodyMediumSecondary),
                      Text(_sourceLabel, style: AppTextStyles.caption),
                    ],
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.edit_note, size: 20),
                  color: AppColors.textSecondary,
                  onPressed: onEditNotes,
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                ),
                const SizedBox(width: 4),
                const Icon(
                  Icons.chevron_right_rounded,
                  size: 20,
                  color: AppColors.textTertiary,
                ),
              ],
            ),
            if (entry.name != null || entry.age != null) ...[
              const SizedBox(height: 2),
              Text(
                [
                  if (entry.name != null) entry.name!,
                  if (entry.age != null) '${entry.age}',
                ].join(', '),
                style: AppTextStyles.caption.copyWith(
                  color: AppColors.textSecondary,
                ),
              ),
            ],
            const SizedBox(height: 10),
            Wrap(
              spacing: 10,
              runSpacing: 8,
              children: [
                OutlinedButton.icon(
                  onPressed: entry.audioFilePath.isEmpty
                      ? null
                      : (isPlaying ? onPause : onPlay),
                  icon: Icon(
                    isPlaying ? Icons.pause_rounded : Icons.play_arrow_rounded,
                    size: 18,
                  ),
                  label: Text(
                    isPlaying
                        ? AppStrings.get('pause', language)
                        : AppStrings.get('listen', language),
                  ),
                ),
                OutlinedButton.icon(
                  onPressed: entry.audioFilePath.isEmpty ? null : onRerun,
                  icon: const Icon(Icons.replay_rounded, size: 18),
                  label: Text(AppStrings.get('run_again', language)),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// ── Notes editor bottom sheet ─────────────────────────────────────────────────

class _NotesEditor extends StatefulWidget {
  final CheckRecord record;
  final String language;
  final Future<void> Function(CheckRecord updated) onSave;

  const _NotesEditor({
    required this.record,
    required this.language,
    required this.onSave,
  });

  @override
  State<_NotesEditor> createState() => _NotesEditorState();
}

class _NotesEditorState extends State<_NotesEditor> {
  late final TextEditingController _nameController;
  late final TextEditingController _ageController;
  late final TextEditingController _notesController;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController(text: widget.record.name ?? '');
    _ageController = TextEditingController(
      text: widget.record.age != null ? widget.record.age.toString() : '',
    );
    _notesController = TextEditingController(text: widget.record.notes ?? '');
  }

  @override
  void dispose() {
    _nameController.dispose();
    _ageController.dispose();
    _notesController.dispose();
    super.dispose();
  }

  String _s(String key) => AppStrings.get(key, widget.language);

  Future<void> _handleSave() async {
    if (_saving) return;
    setState(() => _saving = true);
    final nameTrimmed = _nameController.text.trim();
    final ageTrimmed = _ageController.text.trim();
    final notesTrimmed = _notesController.text.trim();
    final updated = widget.record.copyWith(
      name: nameTrimmed.isEmpty ? null : nameTrimmed,
      age: int.tryParse(ageTrimmed),
      notes: notesTrimmed.isEmpty ? null : notesTrimmed,
      clearName: nameTrimmed.isEmpty,
      clearAge: ageTrimmed.isEmpty,
      clearNotes: notesTrimmed.isEmpty,
    );
    await widget.onSave(updated);
    if (mounted) Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      initialChildSize: 0.55,
      maxChildSize: 0.95,
      minChildSize: 0.3,
      expand: false,
      builder: (context, scrollController) {
        return Container(
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Column(
            children: [
              // Handle
              Padding(
                padding: const EdgeInsets.only(top: 12, bottom: 4),
                child: Center(
                  child: Container(
                    width: 36,
                    height: 4,
                    decoration: BoxDecoration(
                      color: AppColors.border,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),
              ),
              // Header
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 8,
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        _s('edit_notes'),
                        style: AppTextStyles.headingMedium,
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.close_rounded),
                      color: AppColors.textSecondary,
                      onPressed: () => Navigator.pop(context),
                    ),
                  ],
                ),
              ),
              const Divider(height: 1),
              // Fields
              Expanded(
                child: ListView(
                  controller: scrollController,
                  padding: EdgeInsets.fromLTRB(
                    20,
                    20,
                    20,
                    MediaQuery.of(context).viewInsets.bottom + 20,
                  ),
                  children: [
                    Text(_s('name_label'), style: AppTextStyles.labelLarge),
                    const SizedBox(height: 8),
                    TextField(
                      controller: _nameController,
                      decoration: InputDecoration(
                        hintText: _s('name_hint'),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      textCapitalization: TextCapitalization.words,
                    ),
                    const SizedBox(height: 16),
                    Text(_s('age_label'), style: AppTextStyles.labelLarge),
                    const SizedBox(height: 8),
                    TextField(
                      controller: _ageController,
                      decoration: InputDecoration(
                        hintText: _s('age_hint'),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      keyboardType: TextInputType.number,
                    ),
                    const SizedBox(height: 16),
                    Text(_s('notes_label'), style: AppTextStyles.labelLarge),
                    const SizedBox(height: 8),
                    TextField(
                      controller: _notesController,
                      decoration: InputDecoration(
                        hintText: _s('notes_hint'),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        alignLabelWithHint: true,
                      ),
                      maxLines: 3,
                      minLines: 3,
                      textCapitalization: TextCapitalization.sentences,
                    ),
                    const SizedBox(height: 24),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: _saving ? null : _handleSave,
                        child: _saving
                            ? const SizedBox(
                                height: 18,
                                width: 18,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                ),
                              )
                            : Text(_s('save')),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
