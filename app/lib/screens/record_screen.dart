import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../l10n/app_strings.dart';
import '../navigation/app_router.dart';
import '../services/audio_archive_service.dart';
import '../services/audio_service.dart';
import '../services/language_preference_service.dart';
import '../services/voice_capture_quality.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

enum _RecordStage { ambient, vowel, rhythm, speech, processing, complete }

class _TaskConfig {
  const _TaskConfig({
    required this.stage,
    required this.index,
    required this.titleKey,
    required this.promptKey,
    required this.detailKey,
    required this.minimumDuration,
    required this.stopKey,
  });

  final _RecordStage stage;
  final int index;
  final String titleKey;
  final String promptKey;
  final String detailKey;
  final Duration minimumDuration;
  final String stopKey;
}

const _tasks = [
  _TaskConfig(
    stage: _RecordStage.vowel,
    index: 1,
    titleKey: 'task_1_of_3',
    promptKey: 'vowel_prompt',
    detailKey: 'vowel_detail',
    minimumDuration: Duration(seconds: 5),
    stopKey: 'vowel_stop',
  ),
  _TaskConfig(
    stage: _RecordStage.rhythm,
    index: 2,
    titleKey: 'task_2_of_3',
    promptKey: 'rhythm_prompt',
    detailKey: 'rhythm_detail',
    minimumDuration: Duration(seconds: 7),
    stopKey: 'rhythm_stop',
  ),
  _TaskConfig(
    stage: _RecordStage.speech,
    index: 3,
    titleKey: 'task_3_of_3',
    promptKey: 'speech_prompt',
    detailKey: 'speech_detail',
    minimumDuration: Duration(seconds: 10),
    stopKey: 'speech_stop',
  ),
];

class RecordScreen extends StatefulWidget {
  const RecordScreen({super.key});

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  static const _voicePresenceThreshold = 0.12;

  String _language = LanguagePreferenceService.defaultLanguage;

  String _s(String key) => AppStrings.get(key, _language);

  String get _noVoiceErrorMessage => _s('no_voice_error');

  _RecordStage _stage = _RecordStage.ambient;
  Timer? _ticker;
  DateTime? _recordingStartedAt;
  DateTime? _lastPollAt;
  Duration _elapsed = Duration.zero;
  Duration _detectedVoiceDuration = Duration.zero;
  bool _isRecording = false;
  bool _isBusy = false;
  bool _levelRequestInFlight = false;
  double _peakDetectedLevel = 0.0;
  double _smoothedLevel = 0.0;
  String? _statusText;
  String? _errorText;
  int _activeRhythmBeat = -1;
  DateTime? _lastBeatAt;
  final List<double> _waveform = List<double>.filled(48, 0.02);
  final Map<_RecordStage, Float32List> _recordings = {};

  @override
  void initState() {
    super.initState();
    unawaited(_loadLanguage());
    unawaited(_startAmbientMonitor());
  }

  Future<void> _loadLanguage() async {
    final lang = await LanguagePreferenceService.load();
    if (!mounted) return;
    setState(() => _language = lang);
  }

  @override
  void dispose() {
    _ticker?.cancel();
    if (_isRecording) {
      unawaited(_stopRecording(discard: true));
    }
    super.dispose();
  }

  Future<void> _startAmbientMonitor() async {
    setState(() {
      _stage = _RecordStage.ambient;
      _statusText = _s('checking_environment');
      _errorText = null;
      _detectedVoiceDuration = Duration.zero;
      _peakDetectedLevel = 0.0;
    });
    await _beginRecordingSession();
  }

  Future<void> _beginTask(_TaskConfig task) async {
    setState(() {
      _stage = task.stage;
      _statusText = _s('listening');
      _errorText = null;
      _elapsed = Duration.zero;
      _detectedVoiceDuration = Duration.zero;
      _peakDetectedLevel = 0.0;
      _smoothedLevel = 0.0;
      _activeRhythmBeat = -1;
      for (var i = 0; i < _waveform.length; i++) {
        _waveform[i] = 0.02;
      }
    });
    await _beginRecordingSession();
    await HapticFeedback.selectionClick();
  }

  Future<void> _beginRecordingSession() async {
    if (_isBusy) return;
    _ticker?.cancel();
    if (_isRecording) {
      await _stopRecording(discard: true);
    }

    setState(() {
      _isBusy = true;
      _errorText = null;
    });

    try {
      await startRecording();
      final now = DateTime.now();
      _recordingStartedAt = now;
      _lastPollAt = now;
      _isRecording = true;
      _ticker = Timer.periodic(
        const Duration(milliseconds: 80),
        (_) => unawaited(_pollAudioLevel()),
      );
    } on AudioPermissionException catch (_) {
      if (mounted) {
        showDialog<void>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: Text(_s('mic_permission_title')),
            content: Text(_s('mic_permission_denied')),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx),
                child: Text(_s('ok')),
              ),
            ],
          ),
        );
      }
    } on AudioHardwareException catch (e) {
      _errorText = e.message;
    } finally {
      if (mounted) {
        setState(() {
          _isBusy = false;
        });
      }
    }
  }

  Future<void> _pollAudioLevel() async {
    if (!_isRecording || !mounted) return;

    final startedAt = _recordingStartedAt;
    final now = DateTime.now();
    final elapsed = startedAt == null
        ? Duration.zero
        : now.difference(startedAt);
    final tickDuration = _lastPollAt == null
        ? const Duration(milliseconds: 80)
        : now.difference(_lastPollAt!);
    _lastPollAt = now;

    setState(() {
      _elapsed = elapsed;
    });

    if (_levelRequestInFlight) return;
    _levelRequestInFlight = true;

    try {
      final level = await getAudioLevel();
      final normalized = (level * 5.5).clamp(0.0, 1.0);
      if (!mounted) return;
      setState(() {
        if (_stage != _RecordStage.ambient && _stage != _RecordStage.complete) {
          _peakDetectedLevel = math.max(_peakDetectedLevel, normalized);
          if (normalized >= _voicePresenceThreshold) {
            _detectedVoiceDuration += tickDuration;
            if (_errorText == _noVoiceErrorMessage) {
              _errorText = null;
            }
          }
        }
        _smoothedLevel = (_smoothedLevel * 0.7) + (normalized * 0.3);
        _pushWaveformValue(math.max(0.02, normalized));
        _updateRhythmBeat(normalized);
      });
    } catch (_) {
      // Keep UI responsive; next tick will retry.
    } finally {
      _levelRequestInFlight = false;
    }
  }

  void _pushWaveformValue(double value) {
    _waveform.removeAt(0);
    _waveform.add(value);
  }

  void _updateRhythmBeat(double level) {
    if (_stage != _RecordStage.rhythm || level < 0.24) return;
    final now = DateTime.now();
    if (_lastBeatAt != null &&
        now.difference(_lastBeatAt!) < const Duration(milliseconds: 180)) {
      return;
    }
    _lastBeatAt = now;
    _activeRhythmBeat = (_activeRhythmBeat + 1) % 6;
  }

  Future<void> _advanceFromAmbient() async {
    await _stopRecording(discard: true);
    await _beginTask(_tasks.first);
  }

  Future<void> _finishCurrentTask() async {
    if (!_hasMinimumVoiceActivity) {
      final currentTask = _tasks.firstWhere((task) => task.stage == _stage);
      await _stopRecording(discard: true);
      if (!mounted) return;
      setState(() {
        _errorText = _noVoiceErrorMessage;
        _statusText = null;
        _elapsed = Duration.zero;
        _detectedVoiceDuration = Duration.zero;
        _peakDetectedLevel = 0.0;
        _smoothedLevel = 0.0;
        _activeRhythmBeat = -1;
        _lastBeatAt = null;
        for (var i = 0; i < _waveform.length; i++) {
          _waveform[i] = 0.02;
        }
      });
      await HapticFeedback.heavyImpact();
      await Future<void>.delayed(const Duration(milliseconds: 900));
      if (!mounted) return;
      await _beginTask(currentTask);
      return;
    }

    final currentTask = _tasks.firstWhere((task) => task.stage == _stage);
    final recording = await _stopRecording(discard: false);
    if (recording == null) return;

    _recordings[_stage] = recording;
    setState(() {
      _statusText = switch (_stage) {
        _RecordStage.vowel =>
          '${_s('vowel_complete')} ${_formatSeconds(_elapsed)}s',
        _RecordStage.rhythm => _s('rhythm_complete'),
        _RecordStage.speech => _s('speech_complete'),
        _ => _statusText,
      };
    });
    await HapticFeedback.mediumImpact();

    await Future<void>.delayed(const Duration(milliseconds: 1200));
    if (!mounted) return;

    final nextIndex = currentTask.index;
    if (nextIndex < _tasks.length) {
      await _beginTask(_tasks[nextIndex]);
      return;
    }

    setState(() {
      _stage = _RecordStage.processing;
      _statusText = _s('analyzing_voice');
    });
    await Future<void>.delayed(const Duration(milliseconds: 700));
    if (!mounted) return;
    final taskPcmList = <Float32List>[
      if (_recordings[_RecordStage.vowel] != null)
        _recordings[_RecordStage.vowel]!,
      if (_recordings[_RecordStage.rhythm] != null)
        _recordings[_RecordStage.rhythm]!,
      if (_recordings[_RecordStage.speech] != null)
        _recordings[_RecordStage.speech]!,
    ];
    final taskSampleLengths = taskPcmList.map((pcm) => pcm.length).toList();
    final combined = <double>[
      ...?_recordings[_RecordStage.vowel],
      ...?_recordings[_RecordStage.rhythm],
      ...?_recordings[_RecordStage.speech],
    ];
    final recordedPcm = Float32List.fromList(combined);
    final wavBytes = AudioArchiveService.encodePcm16Wav(recordedPcm, 16000);
    Navigator.of(context).pushReplacementNamed(
      AppRoutes.analysis,
      arguments: AnalysisArguments(
        recordedPcm: recordedPcm,
        sampleRate: 16000,
        referenceWavBytes: wavBytes,
        taskPcmList: taskPcmList,
        taskSampleLengths: taskSampleLengths,
      ),
    );
  }

  Future<Float32List?> _stopRecording({required bool discard}) async {
    _ticker?.cancel();
    _ticker = null;
    if (!_isRecording) return null;

    try {
      final pcm = await stopRecording();
      _isRecording = false;
      _recordingStartedAt = null;
      _lastPollAt = null;
      return discard ? null : pcm;
    } on AudioHardwareException catch (e) {
      if (mounted) {
        setState(() => _errorText = e.message);
      }
      _isRecording = false;
      return null;
    }
  }

  bool get _ambientIsQuiet => _smoothedLevel < 0.18;

  bool get _canStopCurrentTask {
    final task = _tasks.where((task) => task.stage == _stage).firstOrNull;
    if (task == null || _elapsed < task.minimumDuration || !_isRecording) {
      return false;
    }
    if (_isBusy) return false;
    if (task.stage == _RecordStage.speech && !_hasMinimumVoiceActivity) {
      return false;
    }
    return true;
  }

  Duration _requiredVoiceDuration(_RecordStage stage) => switch (stage) {
    _RecordStage.vowel => const Duration(milliseconds: 700),
    _RecordStage.rhythm => const Duration(milliseconds: 900),
    _RecordStage.speech => const Duration(milliseconds: 1200),
    _ => Duration.zero,
  };

  bool get _hasMinimumVoiceActivity {
    return _voiceCaptureQuality.hasEnoughVoice;
  }

  VoiceCaptureQualityResult get _voiceCaptureQuality {
    return VoiceCaptureQuality.evaluate(
      detectedVoiceDuration: _detectedVoiceDuration,
      requiredVoiceDuration: _requiredVoiceDuration(_stage),
      peakDetectedLevel: _peakDetectedLevel,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_s('voice_check')),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () => Navigator.of(context).pop(),
        ),
        actions:
            _stage == _RecordStage.complete || _stage == _RecordStage.ambient
            ? null
            : [
                IconButton(
                  icon: const Icon(Icons.refresh_rounded),
                  tooltip: 'Restart test',
                  onPressed: _isBusy
                      ? null
                      : () => unawaited(_confirmRestartTest()),
                ),
              ],
      ),
      body: SafeArea(
        child: AnimatedSwitcher(
          duration: const Duration(milliseconds: 350),
          child: Padding(
            key: ValueKey(_stage),
            padding: const EdgeInsets.fromLTRB(24, 12, 24, 24),
            child: switch (_stage) {
              _RecordStage.ambient => _buildAmbientStage(),
              _RecordStage.vowel ||
              _RecordStage.rhythm ||
              _RecordStage.speech => _buildTaskStage(
                _tasks.firstWhere((task) => task.stage == _stage),
              ),
              _RecordStage.processing => _buildProcessingStage(),
              _RecordStage.complete => _buildCompletionStage(),
            },
          ),
        ),
      ),
    );
  }

  Widget _buildAmbientStage() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: SingleChildScrollView(
            padding: EdgeInsets.zero,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _FlowHeader(stepLabel: '1 of 3', title: _s('find_quiet_space')),
                const SizedBox(height: 12),
                Text(_s('hold_phone'), style: AppTextStyles.bodyLarge),
                const SizedBox(height: 20),
                _AmbientMeter(
                  level: _smoothedLevel,
                  isQuiet: _ambientIsQuiet,
                  language: _language,
                ),
                const SizedBox(height: 16),
                _AmbientChip(isQuiet: _ambientIsQuiet, language: _language),
                const SizedBox(height: 12),
                _buildStatusCard(),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton(
            onPressed: _isBusy ? null : () => unawaited(_advanceFromAmbient()),
            child: Text(_s('im_ready')),
          ),
        ),
      ],
    );
  }

  Widget _buildTaskStage(_TaskConfig task) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _FlowHeader(stepLabel: '${task.index} of 3', title: _s(task.titleKey)),
        const SizedBox(height: 16),
        Text(_s(task.promptKey), style: AppTextStyles.headingLarge),
        const SizedBox(height: 8),
        Text(_s(task.detailKey), style: AppTextStyles.bodyMediumSecondary),
        const SizedBox(height: 16),
        Expanded(
          child: LayoutBuilder(
            builder: (context, constraints) {
              return SingleChildScrollView(
                physics: const BouncingScrollPhysics(),
                child: ConstrainedBox(
                  constraints: BoxConstraints(minHeight: constraints.maxHeight),
                  child: Center(
                    child: switch (task.stage) {
                      _RecordStage.vowel => _CircularLevelVisualizer(
                        level: _smoothedLevel,
                        elapsed: _elapsed,
                        language: _language,
                        maxDiameter: math.min(
                          228,
                          math.min(
                            constraints.maxWidth * 0.7,
                            constraints.maxHeight * 0.82,
                          ),
                        ),
                      ),
                      _RecordStage.rhythm => _RhythmVisualizer(
                        level: _smoothedLevel,
                        activeBeat: _activeRhythmBeat,
                        language: _language,
                        maxWidth: constraints.maxWidth,
                        maxHeight: constraints.maxHeight,
                      ),
                      _RecordStage.speech => _WaveformVisualizer(
                        levels: _waveform,
                        maxHeight: math.min(180, constraints.maxHeight * 0.72),
                      ),
                      _ => const SizedBox.shrink(),
                    },
                  ),
                ),
              );
            },
          ),
        ),
        const SizedBox(height: 8),
        _TaskProgress(
          elapsed: _elapsed,
          minimumDuration: task.minimumDuration,
          caption: task.stage == _RecordStage.speech
              ? _s(
                  'minimum_duration',
                ).replaceAll('{n}', '${task.minimumDuration.inSeconds}')
              : _s(
                  'minimum_before_stop',
                ).replaceAll('{n}', '${task.minimumDuration.inSeconds}'),
        ),
        if (task.stage == _RecordStage.speech) ...[
          const SizedBox(height: 8),
          _VoiceCaptureCue(
            quality: _voiceCaptureQuality,
            minimumDurationComplete: _elapsed >= task.minimumDuration,
            language: _language,
          ),
        ],
        const SizedBox(height: 8),
        _buildStatusCard(),
        const SizedBox(height: 8),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: _canStopCurrentTask
                ? () => unawaited(_finishCurrentTask())
                : null,
            icon: const Icon(Icons.stop_circle_outlined),
            label: Text(_s(task.stopKey)),
          ),
        ),
        SizedBox(
          height: 32,
          child: TextButton(
            onPressed: _isBusy ? null : () => unawaited(_confirmRestartTest()),
            child: Text(_s('restart_test')),
          ),
        ),
      ],
    );
  }

  Widget _buildCompletionStage() {
    return Column(
      children: [
        const Spacer(),
        Container(
          width: 112,
          height: 112,
          decoration: BoxDecoration(
            color: AppColors.primary.withAlpha(18),
            shape: BoxShape.circle,
          ),
          child: const Icon(
            Icons.check_rounded,
            size: 52,
            color: AppColors.primary,
          ),
        ),
        const SizedBox(height: 28),
        Text(
          _s('voice_tasks_complete'),
          style: AppTextStyles.displayMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 12),
        Text(
          _statusText ?? _s('samples_ready'),
          style: AppTextStyles.bodyLarge,
          textAlign: TextAlign.center,
        ),
        const Spacer(),
      ],
    );
  }

  Widget _buildProcessingStage() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        TweenAnimationBuilder<double>(
          tween: Tween(begin: 0.88, end: 1),
          duration: const Duration(milliseconds: 900),
          curve: Curves.easeInOut,
          builder: (context, value, child) =>
              Transform.scale(scale: value, child: child),
          child: Container(
            width: 164,
            height: 164,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: AppColors.primary.withAlpha(16),
              border: Border.all(
                color: AppColors.primary.withAlpha(48),
                width: 2,
              ),
            ),
            child: const Center(
              child: SizedBox(
                width: 46,
                height: 46,
                child: CircularProgressIndicator(
                  strokeWidth: 3,
                  color: AppColors.primary,
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 28),
        Text(
          _s('analyzing_voice'),
          style: AppTextStyles.headingLarge,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 10),
        Text(
          _s('preparing_summary'),
          style: AppTextStyles.bodyMediumSecondary,
          textAlign: TextAlign.center,
        ),
      ],
    );
  }

  Widget _buildStatusCard() {
    final ambientMessage = _ambientIsQuiet
        ? null
        : _s('background_noise_warning');
    final message =
        _errorText ??
        (_stage == _RecordStage.ambient ? ambientMessage : _statusText);
    if (_stage != _RecordStage.ambient &&
        _errorText == null &&
        message == _s('listening')) {
      return const SizedBox.shrink();
    }
    if (message == null) return const SizedBox.shrink();
    final isError = _errorText != null;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        color: isError
            ? AppColors.riskElevatedContainer
            : AppColors.surfaceElevated,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: isError
              ? AppColors.riskElevated.withAlpha(48)
              : AppColors.border,
        ),
      ),
      child: Text(
        message,
        style: AppTextStyles.bodyMedium.copyWith(
          color: isError ? AppColors.riskElevated : AppColors.textSecondary,
        ),
      ),
    );
  }

  String _formatSeconds(Duration value) {
    final seconds = value.inMilliseconds / 1000;
    return seconds.toStringAsFixed(1);
  }

  Future<void> _confirmRestartTest() async {
    if (!mounted) return;
    await showModalBottomSheet<void>(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (sheetContext) => Padding(
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
            Text(_s('restart_voice_check'), style: AppTextStyles.headingMedium),
            const SizedBox(height: 10),
            Text(
              _s('restart_confirmation'),
              style: AppTextStyles.bodyMediumSecondary,
            ),
            const SizedBox(height: 20),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.of(sheetContext).pop();
                  unawaited(_restartTest());
                },
                child: Text(_s('restart_test')),
              ),
            ),
            TextButton(
              onPressed: () => Navigator.of(sheetContext).pop(),
              child: Text(_s('cancel')),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _restartTest() async {
    setState(() {
      _statusText = null;
      _errorText = null;
      _elapsed = Duration.zero;
      _detectedVoiceDuration = Duration.zero;
      _peakDetectedLevel = 0.0;
      _smoothedLevel = 0.0;
      _activeRhythmBeat = -1;
      _lastBeatAt = null;
      for (var i = 0; i < _waveform.length; i++) {
        _waveform[i] = 0.02;
      }
      _recordings.clear();
    });
    if (_isRecording) {
      await _stopRecording(discard: true);
    }
    await _startAmbientMonitor();
  }
}

class _FlowHeader extends StatelessWidget {
  const _FlowHeader({required this.stepLabel, required this.title});

  final String stepLabel;
  final String title;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 12,
      runSpacing: 10,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: [
        ConstrainedBox(
          constraints: const BoxConstraints(minWidth: 0, maxWidth: 240),
          child: Text(title, style: AppTextStyles.headingMedium),
        ),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: AppColors.border),
          ),
          child: Text(stepLabel, style: AppTextStyles.captionStrong),
        ),
      ],
    );
  }
}

class _AmbientMeter extends StatelessWidget {
  const _AmbientMeter({
    required this.level,
    required this.isQuiet,
    required this.language,
  });

  final double level;
  final bool isQuiet;
  final String language;

  @override
  Widget build(BuildContext context) {
    final color = isQuiet ? AppColors.riskLow : AppColors.riskModerate;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: 8,
            runSpacing: 6,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              Icon(Icons.graphic_eq_rounded, color: color),
              Text(
                AppStrings.get('ambient_level', language),
                style: AppTextStyles.labelLarge.copyWith(color: color),
              ),
            ],
          ),
          const SizedBox(height: 14),
          ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: LinearProgressIndicator(
              minHeight: 14,
              value: level.clamp(0.0, 1.0),
              color: color,
              backgroundColor: AppColors.surfaceElevated,
            ),
          ),
          const SizedBox(height: 10),
          Text(
            isQuiet
                ? AppStrings.get('env_quiet', language)
                : AppStrings.get('env_noisy', language),
            style: AppTextStyles.bodyMediumSecondary,
          ),
        ],
      ),
    );
  }
}

class _AmbientChip extends StatelessWidget {
  const _AmbientChip({required this.isQuiet, required this.language});

  final bool isQuiet;
  final String language;

  @override
  Widget build(BuildContext context) {
    final color = isQuiet ? AppColors.riskLow : AppColors.riskModerate;
    final bgColor = isQuiet
        ? AppColors.riskLowContainer
        : AppColors.riskModerateContainer;
    final label = isQuiet
        ? AppStrings.get('env_quiet_chip', language)
        : AppStrings.get('env_noisy_chip', language);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withAlpha(56)),
      ),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 10),
          Expanded(child: Text(label, style: AppTextStyles.bodyMedium)),
        ],
      ),
    );
  }
}

class _TaskProgress extends StatelessWidget {
  const _TaskProgress({
    required this.elapsed,
    required this.minimumDuration,
    required this.caption,
  });

  final Duration elapsed;
  final Duration minimumDuration;
  final String caption;

  @override
  Widget build(BuildContext context) {
    final progress = (elapsed.inMilliseconds / minimumDuration.inMilliseconds)
        .clamp(0.0, 1.0);
    final elapsedLabel =
        '${(elapsed.inMilliseconds / 1000).toStringAsFixed(1)}s';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 12,
          runSpacing: 6,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            Text(elapsedLabel, style: AppTextStyles.headingSmall),
            Text(caption, style: AppTextStyles.caption),
          ],
        ),
        const SizedBox(height: 10),
        ClipRRect(
          borderRadius: BorderRadius.circular(999),
          child: LinearProgressIndicator(
            minHeight: 10,
            value: progress,
            backgroundColor: AppColors.surfaceElevated,
          ),
        ),
      ],
    );
  }
}

class _VoiceCaptureCue extends StatelessWidget {
  const _VoiceCaptureCue({
    required this.quality,
    required this.minimumDurationComplete,
    required this.language,
  });

  final VoiceCaptureQualityResult quality;
  final bool minimumDurationComplete;
  final String language;

  @override
  Widget build(BuildContext context) {
    final isReady = quality.hasEnoughVoice && minimumDurationComplete;
    final color = isReady
        ? AppColors.riskLow
        : quality.hasEnoughVoice
        ? AppColors.primary
        : AppColors.riskModerate;
    final message = !quality.hasEnoughVoice
        ? AppStrings.get('voice_capture_keep_speaking', language)
        : minimumDurationComplete
        ? AppStrings.get('voice_capture_ready', language)
        : AppStrings.get('voice_capture_hold_until_timer', language);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        color: isReady
            ? AppColors.riskLowContainer
            : quality.hasEnoughVoice
            ? AppColors.primaryContainer
            : AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withAlpha(56)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                isReady
                    ? Icons.check_circle_rounded
                    : quality.hasEnoughVoice
                    ? Icons.hourglass_bottom_rounded
                    : Icons.graphic_eq_rounded,
                color: color,
                size: 20,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  AppStrings.get('voice_capture_label', language),
                  style: AppTextStyles.labelLarge.copyWith(color: color),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: LinearProgressIndicator(
              minHeight: 10,
              value: quality.progress,
              color: color,
              backgroundColor: AppColors.surfaceElevated,
            ),
          ),
          const SizedBox(height: 8),
          Text(message, style: AppTextStyles.bodyMediumSecondary),
        ],
      ),
    );
  }
}

class _CircularLevelVisualizer extends StatelessWidget {
  const _CircularLevelVisualizer({
    required this.level,
    required this.elapsed,
    required this.language,
    required this.maxDiameter,
  });

  final double level;
  final Duration elapsed;
  final String language;
  final double maxDiameter;

  @override
  Widget build(BuildContext context) {
    final outerDiameter = math.max(132.0, maxDiameter);
    final innerDiameter = outerDiameter * 0.72;
    final intensity = 0.55 + (level * 0.45);
    final compact = outerDiameter < 170;
    return SizedBox(
      width: outerDiameter,
      height: outerDiameter,
      child: Stack(
        alignment: Alignment.center,
        children: [
          Container(
            width: (outerDiameter * 0.9) * intensity,
            height: (outerDiameter * 0.9) * intensity,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: AppColors.primary.withAlpha(18),
            ),
          ),
          Container(
            width: innerDiameter,
            height: innerDiameter,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(
                color: AppColors.primary.withAlpha(70),
                width: 2,
              ),
              color: AppColors.surface,
              boxShadow: const [
                BoxShadow(
                  color: AppColors.shadowSoft,
                  blurRadius: 24,
                  offset: Offset(0, 12),
                ),
              ],
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  '${(elapsed.inMilliseconds / 1000).toStringAsFixed(1)}s',
                  style: compact
                      ? AppTextStyles.headingMedium
                      : AppTextStyles.displayMedium,
                ),
                SizedBox(height: compact ? 4 : 8),
                Text(
                  AppStrings.get('steady_voice', language),
                  style: compact
                      ? AppTextStyles.caption
                      : AppTextStyles.bodyMediumSecondary,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _RhythmVisualizer extends StatelessWidget {
  const _RhythmVisualizer({
    required this.level,
    required this.activeBeat,
    required this.language,
    required this.maxWidth,
    required this.maxHeight,
  });

  final double level;
  final int activeBeat;
  final String language;
  final double maxWidth;
  final double maxHeight;

  @override
  Widget build(BuildContext context) {
    const syllables = ['pa', 'ta', 'ka', 'pa', 'ta', 'ka'];
    final tileSize = math.max(54.0, math.min(72.0, (maxWidth - 44) / 3));
    final compact = maxHeight < 260;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Wrap(
          alignment: WrapAlignment.center,
          spacing: compact ? 10 : 14,
          runSpacing: compact ? 10 : 14,
          children: List.generate(syllables.length, (index) {
            final isActive = index == activeBeat;
            return AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              width: tileSize,
              height: tileSize,
              decoration: BoxDecoration(
                color: isActive
                    ? AppColors.primaryContainer
                    : AppColors.surface,
                borderRadius: BorderRadius.circular(22),
                border: Border.all(
                  color: isActive ? AppColors.primary : AppColors.border,
                ),
                boxShadow: isActive
                    ? const [
                        BoxShadow(
                          color: AppColors.shadowSoft,
                          blurRadius: 18,
                          offset: Offset(0, 10),
                        ),
                      ]
                    : const [],
              ),
              child: Center(
                child: Text(
                  syllables[index],
                  style:
                      (compact
                              ? AppTextStyles.labelLarge
                              : AppTextStyles.headingMedium)
                          .copyWith(
                            color: isActive
                                ? AppColors.primaryDark
                                : AppColors.textPrimary,
                          ),
                ),
              ),
            );
          }),
        ),
        SizedBox(height: compact ? 14 : 24),
        Text(
          level > 0.24
              ? AppStrings.get('rhythm_detected', language)
              : AppStrings.get('keep_even_pace', language),
          style: AppTextStyles.bodyMediumSecondary,
        ),
      ],
    );
  }
}

class _WaveformVisualizer extends StatelessWidget {
  const _WaveformVisualizer({required this.levels, required this.maxHeight});

  final List<double> levels;
  final double maxHeight;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(maxWidth: 320),
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: AppColors.border),
      ),
      child: SizedBox(
        height: math.max(120, maxHeight),
        child: CustomPaint(
          painter: _WaveformPainter(levels),
          size: Size(double.infinity, math.max(120, maxHeight)),
        ),
      ),
    );
  }
}

class _WaveformPainter extends CustomPainter {
  _WaveformPainter(this.levels);

  final List<double> levels;

  @override
  void paint(Canvas canvas, Size size) {
    final centerY = size.height / 2;
    final barWidth = size.width / (levels.length * 1.45);
    final gap = barWidth * 0.45;
    var x = 0.0;

    final linePaint = Paint()
      ..color = AppColors.primary.withAlpha(28)
      ..strokeWidth = 1;
    canvas.drawLine(Offset(0, centerY), Offset(size.width, centerY), linePaint);

    for (final level in levels) {
      final amplitude = math.max(8.0, level * size.height * 0.44);
      final rect = RRect.fromRectAndRadius(
        Rect.fromCenter(
          center: Offset(x + (barWidth / 2), centerY),
          width: barWidth,
          height: amplitude * 2,
        ),
        const Radius.circular(999),
      );
      final paint = Paint()..color = AppColors.primary.withAlpha(160);
      canvas.drawRRect(rect, paint);
      x += barWidth + gap;
    }
  }

  @override
  bool shouldRepaint(covariant _WaveformPainter oldDelegate) =>
      oldDelegate.levels != levels;
}
