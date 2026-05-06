import 'dart:async';
import 'dart:developer' as developer;
import 'dart:io';
import 'dart:math';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../l10n/app_strings.dart';
import '../models/check_record.dart';
import '../navigation/app_router.dart';
import '../services/audio_archive_service.dart';
import '../services/check_history_service.dart';
import '../services/gemma_download_manager.dart';
import '../services/gemma_service.dart';
import '../services/knowledge_base_service.dart';
import '../services/language_preference_service.dart';
import '../services/voice_metric_policy.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../widgets/doctor_guide_sheet.dart';
import '../widgets/gemma_chat_sheet.dart';
import '../widgets/result_education_sheet.dart';

/// Results screen — Gemma-first design.
///
/// The Gemma narrative is the primary experience, shown inline immediately
/// after the risk card. Voice profile bars, language switcher, and debug
/// details follow. No bottom sheet.
class ResultsScreen extends StatefulWidget {
  const ResultsScreen({super.key, this.arguments});

  final ResultsArguments? arguments;

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  bool _didAutoSave = false;

  // Language & Gemma state
  String _language = LanguagePreferenceService.defaultLanguage;
  String? _narrative;
  String? _gemmaError;
  bool _gemmaBusy = false;

  /// Synchronous guard. [_gemmaBusy] is driven by setState and can't be
  /// trusted to reflect in-flight work across async boundaries, so a second
  /// download-manager notifyListeners call can re-enter _generateNarrative
  /// before the first one has awaited loadModel.
  bool _narrativeInFlight = false;
  bool _detailsExpanded = false;
  final _shareKey = GlobalKey();
  bool _isSharing = false;

  @override
  void initState() {
    super.initState();
    unawaited(_autoSaveRun());
    unawaited(_loadLanguageAndInitGemma());
  }

  @override
  void dispose() {
    GemmaDownloadManager.instance.removeListener(_onDownloadUpdate);
    super.dispose();
  }

  // ── Auto-save ───────────────────────────────────────────────────────────────

  Future<void> _autoSaveRun() async {
    if (_didAutoSave ||
        widget.arguments == null ||
        !widget.arguments!.shouldAutoSave) {
      return;
    }
    _didAutoSave = true;
    final args = widget.arguments!;
    final payload = ResultsAudioPayload(
      pcm: args.recordedPcm,
      sampleRate: args.sampleRate,
      wavBytes: args.referenceWavBytes,
    );
    final id =
        '${DateTime.now().millisecondsSinceEpoch}-${Random().nextInt(1 << 32)}';
    final audioFilePath = await AudioArchiveService.saveRunAudio(
      id: id,
      payload: payload,
    );
    await CheckHistoryService.save(
      CheckRecord.fromResultsArguments(
        id: id,
        timestamp: DateTime.now(),
        source: args.source ?? 'analysis',
        audioFilePath: audioFilePath,
        arguments: args,
      ),
    );
  }

  // ── Gemma lifecycle ─────────────────────────────────────────────────────────

  Future<void> _loadLanguageAndInitGemma() async {
    final lang = await LanguagePreferenceService.load();
    if (!mounted) return;
    setState(() => _language = lang);

    // Observe download manager for state changes.
    GemmaDownloadManager.instance.addListener(_onDownloadUpdate);
    _syncFromDownloadManager();
  }

  void _onDownloadUpdate() {
    if (!mounted) return;
    _syncFromDownloadManager();
  }

  void _syncFromDownloadManager() {
    final mgr = GemmaDownloadManager.instance;
    setState(() {});

    if (mgr.state == GemmaModelState.ready &&
        _narrative == null &&
        !_narrativeInFlight) {
      unawaited(_generateNarrative());
    }
    if (mgr.state == GemmaModelState.error) {
      developer.log(
        'Download manager surfaced error',
        name: 'Results',
        error: mgr.error,
      );
      setState(() => _gemmaError = _s('gemma_error_generic'));
    }
  }

  Future<void> _initializeGemma() async {
    setState(() {
      _gemmaError = null;
      _gemmaBusy = true;
    });
    // Re-trigger download manager if it errored.
    if (GemmaDownloadManager.instance.state == GemmaModelState.error) {
      await GemmaDownloadManager.instance.retry();
    } else if (GemmaDownloadManager.instance.state == GemmaModelState.ready) {
      await _generateNarrative();
    }
    if (mounted) setState(() => _gemmaBusy = false);
  }

  Future<void> _downloadAndGenerate() async {
    setState(() {
      _gemmaBusy = true;
      _gemmaError = null;
    });
    // Download manager is already running — just wait for it.
    // If it's not started or errored, kick it.
    final mgr = GemmaDownloadManager.instance;
    if (mgr.state != GemmaModelState.downloading &&
        mgr.state != GemmaModelState.ready) {
      await mgr.retry();
    }
    if (mounted) setState(() => _gemmaBusy = false);
  }

  Future<void> _generateNarrative() async {
    final features = widget.arguments?.featureSummary;
    final riskScore = widget.arguments?.riskScore;
    if (features == null || features.isEmpty || riskScore == null) {
      return; // No data for Gemma — not an error, just skip.
    }
    if (_narrativeInFlight) return;
    _narrativeInFlight = true;

    setState(() {
      _gemmaBusy = true;
      _gemmaError = null;
    });

    try {
      final gemmaLang = LanguagePreferenceService.gemmaNameFor(_language);
      final narrative = await GemmaService.analyze(
        features: features,
        riskScore: riskScore,
        language: gemmaLang,
        modelScores: widget.arguments?.modelScores,
      );
      if (!mounted) return;
      setState(() => _narrative = narrative);
      unawaited(_prefetchEducation());
    } catch (error, stackTrace) {
      if (!mounted) return;
      developer.log(
        'Narrative generation failed',
        name: 'Results',
        error: error,
        stackTrace: stackTrace,
      );
      // If model failed to load, sync download manager so retry triggers re-download.
      if (GemmaService.state == GemmaModelState.notDownloaded) {
        developer.log(
          'Model load failed — triggering re-download',
          name: 'Results',
        );
        unawaited(GemmaDownloadManager.instance.retry());
      }
      setState(() => _gemmaError = _s('gemma_error_generic'));
    } finally {
      _narrativeInFlight = false;
      if (mounted) setState(() => _gemmaBusy = false);
    }
  }

  Future<void> _switchLanguage(String language) async {
    if (language == _language) return;
    await LanguagePreferenceService.save(language);
    setState(() {
      _language = language;
      _narrative = null;
    });
    if (GemmaService.state == GemmaModelState.ready) {
      await _generateNarrative();
    }
  }

  Future<void> _prefetchEducation() async {
    final riskScore = widget.arguments?.riskScore;
    if (riskScore == null || GemmaService.state != GemmaModelState.ready) {
      return;
    }
    await KnowledgeBaseService.loadEducationCardsForRisk(
      widget.arguments?.riskLabel ?? 'low',
      language: _language,
    );
    unawaited(
      GemmaService.generateEducationIntro(
        riskScore: riskScore,
        language: _language,
        modelScores: widget.arguments?.modelScores ?? const {},
      ),
    );
  }

  void _showChat(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => GemmaChatSheet(language: _language),
    );
  }

  void _showBiomarkers(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _BiomarkerTable(
        features: widget.arguments!.featureSummary!,
        language: _language,
      ),
    );
  }

  void _showDoctorGuide(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => DoctorGuideSheet(language: _language),
    );
  }

  void _showEducation(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => ResultEducationSheet(
        language: _language,
        riskLabel: widget.arguments?.riskLabel ?? 'low',
        riskScore: widget.arguments?.riskScore,
        modelScores: widget.arguments?.modelScores,
      ),
    );
  }

  // ── Share ─────────────────────────────────────────────────────────────────

  Future<void> _share(BuildContext buttonContext) async {
    if (_isSharing) return;
    setState(() => _isSharing = true);

    // Capture button position before async work (avoids BuildContext across
    // async gaps lint).
    final box = buttonContext.findRenderObject() as RenderBox?;
    final origin = box != null
        ? box.localToGlobal(Offset.zero) & box.size
        : null;

    try {
      await WidgetsBinding.instance.endOfFrame;
      final boundary =
          _shareKey.currentContext!.findRenderObject()!
              as RenderRepaintBoundary;
      final image = await boundary.toImage(pixelRatio: 3.0);
      final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
      image.dispose();
      final bytes = byteData!.buffer.asUint8List();

      final tempDir = await getTemporaryDirectory();
      final file = File(
        '${tempDir.path}/cognitrace_${DateTime.now().millisecondsSinceEpoch}.png',
      );
      await file.writeAsBytes(bytes);

      // Compose share text in the user's selected language.
      final riskKey = switch (widget.arguments?.riskLabel ?? 'low') {
        'low' => 'risk_low',
        'moderate' => 'risk_moderate',
        _ => 'risk_elevated',
      };
      final parts = <String>[_s('app_title'), _s(riskKey)];
      if (_narrative != null) {
        parts.add('');
        parts.add(_narrative!);
      }
      parts.add('');
      parts.add('${_s('powered_by_gemma')} \u00b7 ${_s('on_device_ai')}');

      await Share.shareXFiles(
        [XFile(file.path)],
        text: parts.join('\n'),
        sharePositionOrigin: origin,
      );

      if (await file.exists()) await file.delete();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('${_s('share')}: $e')));
      }
    } finally {
      if (mounted) setState(() => _isSharing = false);
    }
  }

  // ── Voice profile computation ───────────────────────────────────────────────

  List<({String key, double value})> get _profileItems {
    final features = widget.arguments?.featureSummary;
    if (features == null || features.isEmpty) {
      return const [
        (key: 'steadiness', value: 0.0),
        (key: 'clarity', value: 0.0),
        (key: 'rhythm', value: 0.0),
        (key: 'articulation', value: 0.0),
      ];
    }

    final steadiness = _mean([
      1 - _normalized(features['jitter_local'], max: 0.35),
      1 - _normalized(features['jitter_rap'], max: 0.25),
      1 - _normalized(features['f0_cv'], max: 0.35),
    ]);

    final clarity = _mean([
      1 - _normalized(features['shimmer_local'], max: 0.35),
      1 - _normalized(features['shimmer_apq3'], max: 0.25),
    ]);

    final rhythm = _mean([
      _normalized(features['voiced_fraction'], min: 0.2, max: 0.95),
      1 - _normalized(features['zcr'], max: 0.18),
      _normalized(features['rms_energy'], min: 0.005, max: 0.12),
    ]);

    final articulation = _mean([
      _normalized(features['spec_centroid_mean'], min: 900, max: 2800),
      _normalized(features['spec_bandwidth_mean'], min: 900, max: 2600),
      1 - _normalized(features['spec_slope_mean'], min: -0.0035, max: 0),
    ]);

    return [
      (key: 'steadiness', value: steadiness),
      (key: 'clarity', value: clarity),
      (key: 'rhythm', value: rhythm),
      (key: 'articulation', value: articulation),
    ];
  }

  double _normalized(double? value, {double min = 0, required double max}) {
    if (value == null) return 0;
    if (max == min) return 0;
    return ((value - min) / (max - min)).clamp(0.0, 1.0);
  }

  double _mean(List<double> values) {
    if (values.isEmpty) return 0;
    return values.reduce((a, b) => a + b) / values.length;
  }

  bool get _hasDetailData {
    final args = widget.arguments;
    if (args == null) return false;
    return (args.featureSummary != null && args.featureSummary!.isNotEmpty) ||
        (args.modelScores != null && args.modelScores!.isNotEmpty) ||
        (args.trace != null && args.trace!.isNotEmpty) ||
        args.featureError != null;
  }

  // ── Build ───────────────────────────────────────────────────────────────────

  String _s(String key) => AppStrings.get(key, _language);

  bool get _isLowRisk => (widget.arguments?.riskLabel ?? 'low') == 'low';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_s('results')),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () => Navigator.of(
            context,
          ).pushNamedAndRemoveUntil(AppRoutes.home, (_) => false),
        ),
        actions: [
          Builder(
            builder: (ctx) => TextButton(
              onPressed: _isSharing ? null : () => unawaited(_share(ctx)),
              child: _isSharing
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : Text(_s('share')),
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // ── SHAREABLE CONTENT (captured as screenshot for share) ──
              RepaintBoundary(
                key: _shareKey,
                child: Container(
                  color: AppColors.background,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const SizedBox(height: 4),
                      Text(
                        _s('listened_carefully'),
                        style: AppTextStyles.headingLarge,
                      ),
                      const SizedBox(height: 20),

                      // Risk card
                      _ResultCard(
                        risk: widget.arguments?.riskLabel ?? 'low',
                        riskScore: widget.arguments?.riskScore,
                        language: _language,
                      ),
                      const SizedBox(height: 18),
                      const SizedBox(height: 28),

                      // ── GEMMA SECTION (centerpiece) ──
                      _buildGemmaSectionHeader(),
                      const SizedBox(height: 12),
                      _buildGemmaNarrativeCard(),
                      const SizedBox(height: 8),
                      _buildGemmaAttribution(),
                      if (GemmaService.hasActiveSession) ...[
                        const SizedBox(height: 16),
                        if (_isLowRisk) ...[
                          SizedBox(
                            width: double.infinity,
                            child: OutlinedButton.icon(
                              icon: const Icon(
                                Icons.menu_book_outlined,
                                size: 18,
                              ),
                              label: Text(_s('learn_about_result')),
                              onPressed: () => _showEducation(context),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: AppColors.primary,
                                side: const BorderSide(
                                  color: AppColors.primary,
                                ),
                                padding: const EdgeInsets.symmetric(
                                  vertical: 14,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(height: 10),
                          SizedBox(
                            width: double.infinity,
                            child: OutlinedButton.icon(
                              icon: const Icon(Icons.auto_awesome, size: 18),
                              label: Text(_s('ask_gemma')),
                              onPressed: () => _showChat(context),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: AppColors.primary,
                                side: const BorderSide(
                                  color: AppColors.primary,
                                ),
                                padding: const EdgeInsets.symmetric(
                                  vertical: 14,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(height: 10),
                          SizedBox(
                            width: double.infinity,
                            child: OutlinedButton.icon(
                              icon: const Icon(
                                Icons.medical_information_outlined,
                                size: 18,
                              ),
                              label: Text(_s('prepare_doctor_visit')),
                              onPressed: () => _showDoctorGuide(context),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: AppColors.textSecondary,
                                side: const BorderSide(color: AppColors.border),
                                padding: const EdgeInsets.symmetric(
                                  vertical: 14,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                              ),
                            ),
                          ),
                        ] else ...[
                          SizedBox(
                            width: double.infinity,
                            child: OutlinedButton.icon(
                              icon: const Icon(Icons.auto_awesome, size: 18),
                              label: Text(_s('ask_gemma')),
                              onPressed: () => _showChat(context),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: AppColors.primary,
                                side: const BorderSide(
                                  color: AppColors.primary,
                                ),
                                padding: const EdgeInsets.symmetric(
                                  vertical: 14,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(height: 10),
                          SizedBox(
                            width: double.infinity,
                            child: OutlinedButton.icon(
                              icon: const Icon(
                                Icons.menu_book_outlined,
                                size: 18,
                              ),
                              label: Text(_s('learn_about_result')),
                              onPressed: () => _showEducation(context),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: AppColors.textSecondary,
                                side: const BorderSide(color: AppColors.border),
                                padding: const EdgeInsets.symmetric(
                                  vertical: 14,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(height: 10),
                          SizedBox(
                            width: double.infinity,
                            child: OutlinedButton.icon(
                              icon: const Icon(
                                Icons.medical_information_outlined,
                                size: 18,
                              ),
                              label: Text(_s('prepare_doctor_visit')),
                              onPressed: () => _showDoctorGuide(context),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: AppColors.primary,
                                side: const BorderSide(
                                  color: AppColors.primary,
                                ),
                                padding: const EdgeInsets.symmetric(
                                  vertical: 14,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ],
                      if (widget.arguments?.featureSummary != null) ...[
                        const SizedBox(height: 10),
                        SizedBox(
                          width: double.infinity,
                          child: OutlinedButton.icon(
                            icon: const Icon(
                              Icons.table_chart_outlined,
                              size: 18,
                            ),
                            label: Text(_s('view_biomarkers')),
                            onPressed: () => _showBiomarkers(context),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: AppColors.textSecondary,
                              side: const BorderSide(color: AppColors.border),
                              padding: const EdgeInsets.symmetric(vertical: 14),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),
                          ),
                        ),
                      ],
                      const SizedBox(height: 28),

                      // ── VOICE PROFILE ──
                      Text(
                        _s('voice_profile'),
                        style: AppTextStyles.headingSmall,
                      ),
                      const SizedBox(height: 14),
                      _buildVoiceProfileCard(),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),

              // ── LANGUAGE SWITCHER ──
              _buildLanguageStoryCard(),
              const SizedBox(height: 16),
              Text(_s('read_another_language'), style: AppTextStyles.caption),
              const SizedBox(height: 10),
              _buildLanguageChips(),
              const SizedBox(height: 28),

              // ── EXPANDABLE DETAILS ──
              if (_hasDetailData) ...[
                _buildDetailsToggle(),
                if (_detailsExpanded) ...[
                  const SizedBox(height: 12),
                  _buildDetailCards(),
                ],
                const SizedBox(height: 24),
              ],

              // ── DISCLAIMER ──
              _buildDisclaimer(),
              const SizedBox(height: 20),
              TextButton(
                onPressed: () => Navigator.of(
                  context,
                ).pushNamedAndRemoveUntil(AppRoutes.home, (_) => false),
                child: Text(_s('return_home')),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Gemma section widgets ───────────────────────────────────────────────────

  Widget _buildGemmaSectionHeader() {
    return Row(
      children: [
        const Icon(
          Icons.auto_awesome_rounded,
          size: 20,
          color: AppColors.primary,
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(_s('voice_tells_us'), style: AppTextStyles.headingSmall),
        ),
      ],
    );
  }

  Widget _buildGemmaNarrativeCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surfaceElevated,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.primary.withAlpha(40)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: _buildGemmaContent(),
      ),
    );
  }

  List<Widget> _buildGemmaContent() {
    // Error state
    if (_gemmaError != null) {
      return [
        Text(
          _gemmaError!,
          style: AppTextStyles.bodyMedium.copyWith(
            color: AppColors.riskElevated,
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: OutlinedButton(
            onPressed: _gemmaBusy ? null : () => unawaited(_initializeGemma()),
            child: Text(_s('retry')),
          ),
        ),
      ];
    }

    // Model not downloaded — show download progress from the background manager.
    final mgr = GemmaDownloadManager.instance;
    if (mgr.state == GemmaModelState.downloading) {
      final pct = (mgr.progress * 100).toStringAsFixed(0);
      return [
        Text(
          _s('downloading_gemma').replaceAll('{pct}', pct),
          style: AppTextStyles.bodyMedium,
        ),
        const SizedBox(height: 12),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: mgr.progress > 0 ? mgr.progress : null,
            minHeight: 6,
          ),
        ),
      ];
    }

    if (mgr.state == GemmaModelState.notDownloaded ||
        mgr.state == GemmaModelState.error ||
        GemmaService.state == GemmaModelState.notDownloaded ||
        GemmaService.state == GemmaModelState.error) {
      return [
        Text(_s('download_gemma'), style: AppTextStyles.bodyMedium),
        const SizedBox(height: 14),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: () => unawaited(_downloadAndGenerate()),
            icon: const Icon(Icons.download_rounded, size: 20),
            label: Text(
              mgr.state == GemmaModelState.error
                  ? _s('retry_download')
                  : _s('download_model'),
            ),
          ),
        ),
      ];
    }

    // Loading / generating
    if (_gemmaBusy ||
        GemmaService.state == GemmaModelState.loading ||
        _narrative == null) {
      return [
        Row(
          children: [
            const SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                _s('preparing_explanation'),
                style: AppTextStyles.bodyMedium,
              ),
            ),
          ],
        ),
      ];
    }

    // Narrative ready
    return [
      Text(_narrative!, style: AppTextStyles.bodyMedium.copyWith(height: 1.6)),
    ];
  }

  Widget _buildGemmaAttribution() {
    return Row(
      children: [
        const Icon(
          Icons.auto_awesome_rounded,
          size: 14,
          color: AppColors.textTertiary,
        ),
        const SizedBox(width: 6),
        Flexible(
          child: Text(
            '${_s('powered_by_gemma')} \u00b7 ${_s('on_device_ai')}',
            style: AppTextStyles.caption.copyWith(
              color: AppColors.textTertiary,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildLanguageStoryCard() {
    return _StoryCard(
      icon: Icons.language_rounded,
      title: _s('same_result_your_language_title'),
      body: _s('same_result_your_language_body'),
    );
  }

  // ── Voice profile ───────────────────────────────────────────────────────────

  Widget _buildVoiceProfileCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        children: _profileItems.map((item) {
          return _ProfileBar(label: _s(item.key), value: item.value);
        }).toList(),
      ),
    );
  }

  // ── Language switcher ───────────────────────────────────────────────────────

  Widget _buildLanguageChips() {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: LanguagePreferenceService.displayNames.map((lang) {
        final isSelected = lang == _language;
        return ChoiceChip(
          label: Text(lang),
          selected: isSelected,
          onSelected: _gemmaBusy
              ? null
              : (_) => unawaited(_switchLanguage(lang)),
          selectedColor: AppColors.primaryContainer,
          backgroundColor: AppColors.surface,
          side: BorderSide(
            color: isSelected ? AppColors.primary : AppColors.border,
          ),
          labelStyle: AppTextStyles.caption.copyWith(
            color: isSelected ? AppColors.primary : AppColors.textPrimary,
            fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
          ),
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
          showCheckmark: false,
        );
      }).toList(),
    );
  }

  // ── Expandable details ──────────────────────────────────────────────────────

  Widget _buildDetailsToggle() {
    return GestureDetector(
      onTap: () => setState(() => _detailsExpanded = !_detailsExpanded),
      behavior: HitTestBehavior.opaque,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          children: [
            Icon(
              _detailsExpanded
                  ? Icons.expand_less_rounded
                  : Icons.expand_more_rounded,
              size: 20,
              color: AppColors.textSecondary,
            ),
            const SizedBox(width: 6),
            Text(
              _detailsExpanded ? _s('hide_details') : _s('see_details'),
              style: AppTextStyles.caption.copyWith(
                color: AppColors.textSecondary,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetailCards() {
    return Column(
      children: [
        _StoryCard(
          icon: Icons.table_chart_outlined,
          title: _s('raw_analysis_title'),
          body: _s('raw_analysis_body'),
        ),
        const SizedBox(height: 12),
        if (widget.arguments?.featureSummary != null ||
            widget.arguments?.featureError != null)
          _FeatureSummaryCard(
            arguments: widget.arguments!,
            language: _language,
          ),
        if (widget.arguments?.modelScores != null &&
            widget.arguments!.modelScores!.isNotEmpty) ...[
          const SizedBox(height: 12),
          _ModelScoresCard(
            modelName: widget.arguments?.modelName,
            scores: widget.arguments!.modelScores!,
            language: _language,
          ),
        ],
        if (widget.arguments?.trace != null &&
            widget.arguments!.trace!.isNotEmpty) ...[
          const SizedBox(height: 12),
          _TraceCard(trace: widget.arguments!.trace!, language: _language),
        ],
      ],
    );
  }

  // ── Disclaimer ──────────────────────────────────────────────────────────────

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
            child: Text(_s('disclaimer_short'), style: AppTextStyles.caption),
          ),
        ],
      ),
    );
  }
}

// ── Result card (risk headline) ───────────────────────────────────────────────

class _ResultCard extends StatelessWidget {
  final String risk;
  final double? riskScore;
  final String language;

  const _ResultCard({
    required this.risk,
    this.riskScore,
    required this.language,
  });

  Color get _dotColor => switch (risk) {
    'low' => AppColors.riskLow,
    'moderate' => AppColors.riskModerate,
    _ => AppColors.riskElevated,
  };

  Color get _containerColor => switch (risk) {
    'low' => AppColors.riskLowContainer,
    'moderate' => AppColors.riskModerateContainer,
    _ => AppColors.riskElevatedContainer,
  };

  String get _headlineKey => switch (risk) {
    'low' => 'risk_low',
    'moderate' => 'risk_moderate',
    _ => 'risk_elevated',
  };

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: _containerColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _dotColor.withAlpha(40)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 12,
            height: 12,
            margin: const EdgeInsets.only(top: 5),
            decoration: BoxDecoration(color: _dotColor, shape: BoxShape.circle),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Text(
              AppStrings.get(_headlineKey, language),
              style: AppTextStyles.bodyLarge,
            ),
          ),
        ],
      ),
    );
  }
}

// ── Voice profile bar ─────────────────────────────────────────────────────────

class _ProfileBar extends StatelessWidget {
  final String label;
  final double value;

  const _ProfileBar({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Row(
        children: [
          SizedBox(
            width: 110,
            child: Text(label, style: AppTextStyles.bodyMedium),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(3),
              child: LinearProgressIndicator(
                value: value,
                minHeight: 6,
                color: AppColors.primary,
                backgroundColor: AppColors.border,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _StoryCard extends StatelessWidget {
  const _StoryCard({
    required this.icon,
    required this.title,
    required this.body,
  });

  final IconData icon;
  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 18, color: AppColors.primary),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: AppTextStyles.labelLarge),
                const SizedBox(height: 4),
                Text(
                  body,
                  style: AppTextStyles.bodyMedium.copyWith(
                    color: AppColors.textSecondary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ── Detail cards (hidden behind expandable toggle) ────────────────────────────

class _FeatureSummaryCard extends StatelessWidget {
  const _FeatureSummaryCard({required this.arguments, required this.language});

  static const _displayKeys = [
    'f0_mean',
    'jitter_local',
    'shimmer_local',
    'rms_energy',
    'spec_centroid_mean',
  ];

  final ResultsArguments arguments;
  final String language;

  @override
  Widget build(BuildContext context) {
    final error = arguments.featureError;
    final featureSummary = arguments.featureSummary;

    if (error != null) {
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppColors.riskElevatedContainer,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.riskElevated.withAlpha(48)),
        ),
        child: Text(
          error,
          style: AppTextStyles.bodyMedium.copyWith(
            color: AppColors.riskElevated,
          ),
        ),
      );
    }

    if (featureSummary == null || featureSummary.isEmpty) {
      return const SizedBox.shrink();
    }

    final visibleEntries = _displayKeys
        .where(featureSummary.containsKey)
        .map((key) => MapEntry(key, featureSummary[key]!))
        .toList();

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surfaceElevated,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            AppStrings.get('feature_check', language),
            style: AppTextStyles.headingSmall,
          ),
          const SizedBox(height: 10),
          ...visibleEntries.map(
            (entry) => Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Text(
                '${_labelFor(entry.key)}: ${entry.value.toStringAsFixed(entry.value.abs() >= 10 ? 2 : 4)}',
                style: AppTextStyles.bodyMedium,
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _labelFor(String key) => switch (key) {
    'f0_mean' => 'Average pitch',
    'jitter_local' => 'Pitch stability',
    'shimmer_local' => 'Amplitude variation',
    'rms_energy' => 'Voice energy',
    'spec_centroid_mean' => 'Spectral brightness',
    _ => key,
  };
}

class _ModelScoresCard extends StatelessWidget {
  const _ModelScoresCard({
    required this.modelName,
    required this.scores,
    required this.language,
  });

  final String? modelName;
  final Map<String, double> scores;
  final String language;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surfaceElevated,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            AppStrings.get('model_contributions', language),
            style: AppTextStyles.headingSmall,
          ),
          if (modelName != null) ...[
            const SizedBox(height: 6),
            Text(modelName!, style: AppTextStyles.caption),
          ],
          const SizedBox(height: 10),
          ...scores.entries.map(
            (entry) => Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Text(
                '${_labelFor(entry.key)}: ${entry.value.toStringAsFixed(4)}',
                style: AppTextStyles.bodyMedium,
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _labelFor(String key) => switch (key) {
    'xgb_probability' => 'XGBoost probability',
    'lgb_probability' => 'LightGBM probability',
    'cb_probability' => 'CatBoost probability',
    _ => key,
  };
}

// ── Biomarker table (bottom sheet) ───────────────────────────────────────────

class _BiomarkerTable extends StatelessWidget {
  const _BiomarkerTable({required this.features, required this.language});

  final Map<String, double> features;
  final String language;

  String _s(String key) => AppStrings.get(key, language);

  static const _groups = [
    (
      'Voice Steadiness',
      [
        ('duration_s', 'Recording duration'),
        ('f0_mean', 'Average pitch frequency'),
        ('f0_std', 'Pitch variation'),
        ('f0_cv', 'Pitch consistency'),
        ('f0_min', 'Minimum pitch'),
        ('f0_max', 'Maximum pitch'),
        ('f0_range', 'Pitch range'),
        ('f0_p25', 'Pitch 25th percentile'),
        ('f0_p75', 'Pitch 75th percentile'),
        ('f0_iqr', 'Pitch interquartile range'),
        ('voiced_fraction', 'Voiced frame ratio'),
        ('jitter_local', 'Pitch cycle regularity'),
        ('jitter_rap', 'Pitch smoothness'),
        ('jitter_ppq5', 'Pitch 5-point perturbation'),
      ],
    ),
    (
      'Vocal Clarity',
      [
        ('shimmer_local', 'Volume cycle stability'),
        ('shimmer_apq3', 'Volume smoothness (3-point)'),
        ('shimmer_apq5', 'Volume smoothness (5-point)'),
        ('hnr', 'Harmonics-to-noise ratio'),
      ],
    ),
    (
      'Speech Energy',
      [
        ('rms_energy', 'Root-mean-square loudness'),
        ('log_energy', 'Log-scale power'),
        ('zcr', 'Zero-crossing rate'),
        ('peak_amplitude', 'Peak volume'),
      ],
    ),
    (
      'Spectral Profile',
      [
        ('spec_centroid_mean', 'Average spectral brightness'),
        ('spec_centroid_std', 'Brightness variation'),
        ('spec_bandwidth_mean', 'Frequency spread'),
        ('spec_bandwidth_std', 'Frequency spread variation'),
        ('spec_rolloff_mean', 'High-frequency cutoff'),
        ('spec_rolloff_std', 'Cutoff variation'),
        ('spec_slope_mean', 'Spectral tilt'),
        ('spec_slope_std', 'Tilt variation'),
      ],
    ),
    (
      'MFCC (Spectral Envelope)',
      [
        ('mfcc_0_mean', 'Overall energy'),
        ('mfcc_0_std', 'Energy variation'),
        ('mfcc_1_mean', 'Low/high frequency balance'),
        ('mfcc_1_std', 'Balance variation'),
        ('mfcc_2_mean', 'Spectral shape'),
        ('mfcc_2_std', 'Shape variation'),
        ('mfcc_3_mean', 'Fine spectral detail 1'),
        ('mfcc_3_std', 'Detail 1 variation'),
        ('mfcc_4_mean', 'Fine spectral detail 2'),
        ('mfcc_4_std', 'Detail 2 variation'),
        ('mfcc_5_mean', 'Fine spectral detail 3'),
        ('mfcc_5_std', 'Detail 3 variation'),
        ('mfcc_6_mean', 'Fine spectral detail 4'),
        ('mfcc_6_std', 'Detail 4 variation'),
        ('mfcc_7_mean', 'Fine spectral detail 5'),
        ('mfcc_7_std', 'Detail 5 variation'),
        ('mfcc_8_mean', 'Fine spectral detail 6'),
        ('mfcc_8_std', 'Detail 6 variation'),
        ('mfcc_9_mean', 'Fine spectral detail 7'),
        ('mfcc_9_std', 'Detail 7 variation'),
        ('mfcc_10_mean', 'Fine spectral detail 8'),
        ('mfcc_10_std', 'Detail 8 variation'),
        ('mfcc_11_mean', 'Fine spectral detail 9'),
        ('mfcc_11_std', 'Detail 9 variation'),
        ('mfcc_12_mean', 'Fine spectral detail 10'),
        ('mfcc_12_std', 'Detail 10 variation'),
      ],
    ),
  ];

  Color? _statusColor(String key, double value) {
    if (!VoiceMetricPolicy.hasReferenceThreshold(key)) return null;
    final isHigh = VoiceMetricPolicy.isReferenceHigh(key, value);
    return isHigh ? AppColors.riskModerate : AppColors.riskLow;
  }

  String _formatValue(String key, double value) {
    return VoiceMetricPolicy.formatValue(key, value);
  }

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      initialChildSize: 0.7,
      maxChildSize: 0.95,
      minChildSize: 0.4,
      expand: false,
      builder: (_, scrollController) {
        return Container(
          decoration: const BoxDecoration(
            color: AppColors.background,
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Column(
            children: [
              // ── Drag handle ──
              const SizedBox(height: 10),
              Container(
                width: 36,
                height: 4,
                decoration: BoxDecoration(
                  color: AppColors.border,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              // ── Header ──
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 12, 8),
                child: Row(
                  children: [
                    const Icon(
                      Icons.table_chart_outlined,
                      size: 20,
                      color: AppColors.textSecondary,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _s('view_biomarkers'),
                        style: AppTextStyles.headingSmall,
                      ),
                    ),
                    IconButton(
                      onPressed: () => Navigator.of(context).pop(),
                      icon: const Icon(Icons.close_rounded),
                      color: AppColors.textSecondary,
                      iconSize: 22,
                    ),
                  ],
                ),
              ),
              const Divider(height: 1, color: AppColors.divider),
              // ── Scrollable body ──
              Expanded(
                child: ListView.builder(
                  controller: scrollController,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 12,
                  ),
                  itemCount: _groups.length,
                  itemBuilder: (_, groupIndex) {
                    final (groupName, featureList) = _groups[groupIndex];
                    final rows = <Widget>[
                      if (groupIndex > 0)
                        const Padding(
                          padding: EdgeInsets.only(top: 8, bottom: 4),
                          child: Divider(height: 1, color: AppColors.divider),
                        ),
                      Padding(
                        padding: const EdgeInsets.only(top: 12, bottom: 6),
                        child: Text(
                          groupName,
                          style: AppTextStyles.headingSmall.copyWith(
                            fontSize: 14,
                            color: AppColors.textSecondary,
                          ),
                        ),
                      ),
                    ];

                    for (final (key, description) in featureList) {
                      final value = features[key];
                      if (value == null) continue;
                      final dotColor = _statusColor(key, value);
                      rows.add(
                        Padding(
                          padding: const EdgeInsets.symmetric(vertical: 5),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              // Status dot (8×8) or spacer
                              SizedBox(
                                width: 20,
                                child: dotColor != null
                                    ? Padding(
                                        padding: const EdgeInsets.only(top: 5),
                                        child: Container(
                                          width: 8,
                                          height: 8,
                                          decoration: BoxDecoration(
                                            color: dotColor,
                                            shape: BoxShape.circle,
                                          ),
                                        ),
                                      )
                                    : null,
                              ),
                              // Description + key name
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      description,
                                      style: AppTextStyles.bodyMedium,
                                    ),
                                    Text(
                                      key,
                                      style: AppTextStyles.caption.copyWith(
                                        color: AppColors.textTertiary,
                                        fontSize: 12,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(width: 12),
                              // Value
                              Text(
                                _formatValue(key, value),
                                style: AppTextStyles.bodyMedium.copyWith(
                                  fontFeatures: [
                                    const ui.FontFeature.tabularFigures(),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    }

                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: rows,
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

class _TraceCard extends StatelessWidget {
  const _TraceCard({required this.trace, required this.language});

  final Map<String, double> trace;
  final String language;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surfaceElevated,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            AppStrings.get('processing_trace', language),
            style: AppTextStyles.headingSmall,
          ),
          const SizedBox(height: 10),
          ...trace.entries.map(
            (entry) => Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Text(
                '${_labelFor(entry.key)}: ${_format(entry.key, entry.value)}',
                style: AppTextStyles.bodyMedium,
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _labelFor(String key) => switch (key) {
    'audio_seconds' => 'Audio duration',
    'feature_extraction_ms' => 'Feature extraction',
    'inference_ms' => 'Model inference',
    'total_analysis_ms' => 'Total analysis',
    _ => key,
  };

  String _format(String key, double value) {
    if (key == 'audio_seconds') {
      return '${value.toStringAsFixed(2)} s';
    }
    return '${value.toStringAsFixed(0)} ms';
  }
}
