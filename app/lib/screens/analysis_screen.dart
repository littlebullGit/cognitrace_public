import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../l10n/app_strings.dart';
import '../navigation/app_router.dart';
import '../services/audio_service.dart';
import '../services/gemma_service.dart';
import '../services/inference_service.dart';
import '../services/language_preference_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// Analysis screen — the cinematic sequence while AI processes audio.
///
/// Stages correspond to real pipeline steps (spec §4).
/// In this shell the stages auto-advance with realistic timing.
class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key, this.arguments});

  final AnalysisArguments? arguments;

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen>
    with SingleTickerProviderStateMixin {
  static const _featureExtractionTimeout = Duration(seconds: 20);
  static const _referenceSampleExtractionTimeout = Duration(seconds: 120);
  static const _minimumProcessingDuration = Duration(seconds: 2);
  static const _stageTick = Duration(milliseconds: 900);

  late final AnimationController _orbController;
  Timer? _stageTimer;
  DateTime? _processingStartedAt;
  int _activeStage = 0;
  Map<String, double>? _featureSummary;
  InferenceResult? _inferenceResult;
  Map<String, double>? _trace;
  String? _featureError;
  bool _featureExtractionFinished = false;
  bool _minimumProcessingFinished = false;
  bool _didNavigate = false;
  List<int>? _taskSampleLengths;
  String _language = LanguagePreferenceService.defaultLanguage;

  String _s(String key) => AppStrings.get(key, _language);

  List<String> get _stages => [
    _s('stage_steadiness'),
    _s('stage_clarity'),
    _s('stage_rhythm'),
    _s('stage_patterns'),
    _s('stage_assessment'),
  ];

  @override
  void initState() {
    super.initState();
    _orbController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 8),
    )..repeat();
    _processingStartedAt = DateTime.now();
    unawaited(_loadLanguage());
    unawaited(_loadFeatureSummary());
    _startProcessingAutomation();
  }

  Future<void> _loadLanguage() async {
    final lang = await LanguagePreferenceService.load();
    if (!mounted) return;
    setState(() => _language = lang);
  }

  @override
  void dispose() {
    _stageTimer?.cancel();
    _orbController.dispose();
    super.dispose();
  }

  Future<void> _loadFeatureSummary() async {
    final args = widget.arguments;
    final hasReferenceBytes =
        args?.referenceWavBytes != null && args!.referenceWavBytes!.isNotEmpty;
    final hasRecordedPcm = args != null && args.recordedPcm.isNotEmpty;
    if (args == null || (!hasRecordedPcm && !hasReferenceBytes)) {
      if (!mounted) return;
      setState(() {
        _featureError =
            'No recorded audio was available for feature extraction.';
        _featureExtractionFinished = true;
      });
      _maybeAdvanceToResults();
      return;
    }

    try {
      final extractionTimeout = args.isReferenceSample
          ? _referenceSampleExtractionTimeout
          : _featureExtractionTimeout;
      final totalStopwatch = Stopwatch()..start();
      final extractionStopwatch = Stopwatch()..start();

      final pcmList = args.taskPcmList;
      Map<String, double> features;
      InferenceResult inference;
      Map<String, double> traceMap;
      int extractionMs = 0;
      int inferenceMs = 0;

      if (pcmList != null && pcmList.length > 1) {
        // Per-task extraction and inference (correct path).
        final taskFeatureMaps = <Map<String, double>>[];
        final taskResults = <InferenceResult>[];
        Map<String, double> lastExtTrace = {};

        for (final pcm in pcmList) {
          final ext = await extractFeatures(
            pcm: pcm,
            sampleRate: args.sampleRate,
          ).timeout(extractionTimeout);
          taskFeatureMaps.add(ext.features);
          lastExtTrace = ext.trace;
        }

        extractionStopwatch.stop();
        extractionMs = extractionStopwatch.elapsedMilliseconds;

        // Inference runs after all extractions, timed separately.
        final inferenceStopwatch = Stopwatch()..start();
        for (final featureMap in taskFeatureMaps) {
          final result = await InferenceService.runRiskInference(featureMap);
          taskResults.add(result);
        }
        inferenceStopwatch.stop();
        inferenceMs = inferenceStopwatch.elapsedMilliseconds;

        // Use vowel task features (first task) for display, Gemma, and
        // threshold comparisons.  Vowel features match the training
        // distribution and the sustained-vowel reference thresholds in
        // VoiceMetricPolicy.  The risk score is already correctly averaged
        // from per-task inference above.
        features = Map<String, double>.from(taskFeatureMaps.first);

        // Average risk scores across tasks.
        final avgRisk =
            taskResults.map((r) => r.riskScore).reduce((a, b) => a + b) /
            taskResults.length;
        final avgModelScores = <String, double>{};
        for (final key in taskResults.first.modelScores.keys) {
          avgModelScores[key] =
              taskResults
                  .map((r) => r.modelScores[key] ?? 0)
                  .reduce((a, b) => a + b) /
              taskResults.length;
        }

        inference = InferenceResult(
          riskScore: avgRisk,
          riskLabel: InferenceService.labelFor(avgRisk),
          featureVector: taskResults.last.featureVector,
          modelName: taskResults.first.modelName,
          modelScores: avgModelScores,
        );
        traceMap = lastExtTrace;
        _taskSampleLengths = pcmList.map((p) => p.length).toList();
      } else {
        // Single extraction fallback (rerun from saved WAV, single bundled clip).
        final extraction =
            await (args.referenceWavBytes != null
                    ? extractFeaturesFromWavBytes(
                        wavBytes: args.referenceWavBytes!,
                      )
                    : extractFeatures(
                        pcm: args.recordedPcm,
                        sampleRate: args.sampleRate,
                      ))
                .timeout(extractionTimeout);
        extractionStopwatch.stop();
        extractionMs = extractionStopwatch.elapsedMilliseconds;
        features = extraction.features;
        traceMap = extraction.trace;
        final inferenceStopwatch = Stopwatch()..start();
        inference = await InferenceService.runRiskInference(features);
        inferenceStopwatch.stop();
        inferenceMs = inferenceStopwatch.elapsedMilliseconds;
      }

      totalStopwatch.stop();
      if (!mounted) return;
      setState(() {
        _featureSummary = Map<String, double>.from(features);
      });

      final audioSeconds = pcmList != null && pcmList.length > 1
          ? pcmList.fold<int>(0, (sum, p) => sum + p.length) / args.sampleRate
          : args.recordedPcm.isNotEmpty
          ? args.recordedPcm.length / args.sampleRate
          : (features['duration_s'] ?? 0);
      final trace = {
        'audio_seconds': audioSeconds,
        'feature_extraction_ms': extractionMs.toDouble(),
        'inference_ms': inferenceMs.toDouble(),
        'total_analysis_ms': totalStopwatch.elapsedMilliseconds.toDouble(),
        ...traceMap,
      };
      debugPrint('CogniTrace analysis trace: $trace');
      if (!mounted) return;
      setState(() {
        _inferenceResult = inference;
        _trace = trace;
        _featureExtractionFinished = true;
      });
      // Preload Gemma in the background while the analysis animation plays.
      // By the time the results screen opens, the model is already in memory.
      unawaited(GemmaService.loadModel());
      _maybeAdvanceToResults();
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _featureError = error is TimeoutException
            ? args.isReferenceSample
                  ? 'The full reference sample is still taking too long on this device. We need to optimize the long-sample path further.'
                  : 'On-device feature extraction is taking longer than expected. We saved your recording, but couldn\'t finish the feature check this time.'
            : error.toString();
        _featureExtractionFinished = true;
      });
      _maybeAdvanceToResults();
    }
  }

  void _startProcessingAutomation() {
    _stageTimer = Timer.periodic(_stageTick, (_) {
      if (!mounted || _didNavigate) return;
      setState(() {
        _activeStage = (_activeStage + 1) % _stages.length;
        final startedAt = _processingStartedAt;
        if (startedAt != null &&
            !_minimumProcessingFinished &&
            DateTime.now().difference(startedAt) >=
                _minimumProcessingDuration) {
          _minimumProcessingFinished = true;
        }
      });
      _maybeAdvanceToResults();
    });

    Future<void>.delayed(_minimumProcessingDuration, () {
      if (!mounted || _minimumProcessingFinished) return;
      setState(() {
        _minimumProcessingFinished = true;
      });
      _maybeAdvanceToResults();
    });
  }

  void _maybeAdvanceToResults() {
    if (!mounted ||
        _didNavigate ||
        !_featureExtractionFinished ||
        !_minimumProcessingFinished) {
      return;
    }
    _stageTimer?.cancel();
    _didNavigate = true;
    final source =
        widget.arguments?.sourceTag ??
        (widget.arguments?.isReferenceSample == true
            ? 'bundled_sample'
            : 'analysis');
    Navigator.of(context).pushReplacementNamed(
      AppRoutes.results,
      arguments: ResultsArguments(
        source: source,
        recordedPcm: widget.arguments?.recordedPcm,
        referenceWavBytes: widget.arguments?.referenceWavBytes,
        sampleRate: widget.arguments?.sampleRate,
        shouldAutoSave: widget.arguments?.shouldAutoSave ?? true,
        riskScore: _inferenceResult?.riskScore,
        riskLabel: _inferenceResult?.riskLabel,
        modelName: _inferenceResult?.modelName,
        modelScores: _inferenceResult?.modelScores,
        trace: _trace,
        featureSummary: _featureSummary,
        featureError: _featureError,
        taskSampleLengths:
            _taskSampleLengths ?? widget.arguments?.taskSampleLengths,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Stack(
          children: [
            SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(28, 32, 28, 28),
              child: Column(
                children: [
                  _ProcessingOrb(controller: _orbController),
                  const SizedBox(height: 24),
                  Text(
                    _s('listening_voice'),
                    style: AppTextStyles.displayMedium,
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 28),
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: AppColors.surface,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: AppColors.border),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: List.generate(_stages.length, (i) {
                        final state = i == _activeStage
                            ? _StageState.active
                            : _featureExtractionFinished
                            ? _StageState.complete
                            : _StageState.pending;
                        return _StageRow(
                          label: _stages[i],
                          state: state,
                          isLast: i == _stages.length - 1,
                        );
                      }),
                    ),
                  ),
                  const SizedBox(height: 16),
                  if (kDebugMode)
                    _FeatureDebugCard(
                      featureSummary: _featureSummary,
                      featureError: _featureError,
                      language: _language,
                    ),
                ],
              ),
            ),
            Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.only(top: 8, right: 8),
                child: IconButton(
                  icon: const Icon(Icons.close, color: AppColors.textSecondary),
                  onPressed: () => Navigator.of(context).pop(),
                  tooltip: 'Cancel',
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ProcessingOrb extends StatelessWidget {
  const _ProcessingOrb({required this.controller});

  final AnimationController controller;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 180,
      height: 180,
      child: AnimatedBuilder(
        animation: controller,
        builder: (context, child) {
          return Stack(
            alignment: Alignment.center,
            children: [
              Transform.rotate(
                angle: controller.value * 6.283185307179586,
                child: Container(
                  width: 168,
                  height: 168,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: AppColors.primary.withAlpha(72),
                      width: 2,
                    ),
                  ),
                ),
              ),
              Transform.rotate(
                angle: -controller.value * 9.42477796076938,
                child: SizedBox(
                  width: 136,
                  height: 136,
                  child: CircularProgressIndicator(
                    value: 0.76,
                    strokeWidth: 6,
                    color: AppColors.primary,
                    backgroundColor: AppColors.primaryContainer,
                  ),
                ),
              ),
              Container(
                width: 92,
                height: 92,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: AppColors.primary.withAlpha(20),
                ),
              ),
              const Icon(
                Icons.graphic_eq_rounded,
                color: AppColors.primary,
                size: 34,
              ),
            ],
          );
        },
      ),
    );
  }
}

class _FeatureDebugCard extends StatelessWidget {
  const _FeatureDebugCard({
    required this.featureSummary,
    required this.featureError,
    required this.language,
  });

  final Map<String, double>? featureSummary;
  final String? featureError;
  final String language;

  @override
  Widget build(BuildContext context) {
    if (featureError != null) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppColors.riskElevatedContainer,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.riskElevated.withAlpha(48)),
        ),
        child: Text(
          featureError!,
          style: AppTextStyles.bodyMedium.copyWith(
            color: AppColors.riskElevated,
          ),
        ),
      );
    }

    if (featureSummary == null) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppColors.surfaceElevated,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.border),
        ),
        child: Text(
          AppStrings.get('extracting_features', language),
          style: AppTextStyles.bodyMediumSecondary,
        ),
      );
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surfaceElevated,
        borderRadius: BorderRadius.circular(14),
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
          ...featureSummary!.entries.map(
            (entry) => Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Text(
                '${entry.key}: ${entry.value.toStringAsFixed(entry.value.abs() >= 10 ? 2 : 4)}',
                style: AppTextStyles.bodyMedium.copyWith(
                  color: AppColors.textPrimary,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

enum _StageState { pending, active, complete }

class _StageRow extends StatelessWidget {
  final String label;
  final _StageState state;
  final bool isLast;

  const _StageRow({
    required this.label,
    required this.state,
    this.isLast = false,
  });

  @override
  Widget build(BuildContext context) {
    final (color, icon) = switch (state) {
      _StageState.complete => (AppColors.riskLow, Icons.check_circle_rounded),
      _StageState.active => (AppColors.primary, Icons.radio_button_checked),
      _StageState.pending => (
        AppColors.textTertiary,
        Icons.radio_button_unchecked,
      ),
    };

    return Padding(
      padding: EdgeInsets.only(bottom: isLast ? 0 : 14),
      child: Row(
        children: [
          Icon(icon, size: 20, color: color),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              label,
              style: AppTextStyles.bodyMedium.copyWith(color: color),
            ),
          ),
          if (state == _StageState.active)
            const SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: AppColors.primary,
              ),
            ),
        ],
      ),
    );
  }
}
