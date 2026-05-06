import 'dart:async';

import 'package:flutter/widgets.dart';

import 'gemma_service.dart';

/// App-wide Gemma download manager.
///
/// Starts downloading eagerly on app launch. Home screen and results screen
/// both observe [progress] and [state] to show status without re-triggering.
class GemmaDownloadManager extends ChangeNotifier with WidgetsBindingObserver {
  GemmaDownloadManager._();

  static final instance = GemmaDownloadManager._();

  GemmaModelState _state = GemmaModelState.notDownloaded;
  double _progress = 0.0;
  String? _error;
  DateTime? _downloadStartedAt;
  double _lastLoggedProgress = -1;
  bool _observingLifecycle = false;
  bool _resumeOnForegroundRunning = false;

  GemmaModelState get state => _state;
  double get progress => _progress;
  String? get error => _error;

  bool _started = false;

  String get _elapsed {
    if (_downloadStartedAt == null) return '0s';
    final d = DateTime.now().difference(_downloadStartedAt!);
    if (d.inMinutes > 0) {
      return '${d.inMinutes}m ${d.inSeconds % 60}s';
    }
    return '${d.inSeconds}s';
  }

  /// Call once from main.dart. Initializes Gemma and starts background
  /// download if the model isn't already installed.
  Future<void> ensureStarted() async {
    _ensureLifecycleObserver();
    if (_started) return;
    _started = true;

    debugPrint('[Gemma] Checking for GGUF model on disk…');
    Timer? heartbeat;
    try {
      await GemmaService.initialize();
      _state = GemmaService.state;
      notifyListeners();

      if (_state == GemmaModelState.ready) {
        debugPrint('[Gemma] Model already installed — skipping download.');
        return;
      }

      debugPrint('[Gemma] Model not installed. Starting download…');
      _state = GemmaModelState.downloading;
      _downloadStartedAt = DateTime.now();
      _lastLoggedProgress = -1;
      notifyListeners();

      // Periodic heartbeat so silence doesn't look like a hang.
      heartbeat = Timer.periodic(const Duration(seconds: 30), (_) {
        final pct = (_progress * 100).toStringAsFixed(1);
        debugPrint('[Gemma] ⏳ Still downloading… $pct% ($_elapsed)');
      });

      await GemmaService.downloadModel(
        onProgress: (p) {
          _progress = p;
          notifyListeners();
          _maybeLogProgress(p);
        },
      );

      heartbeat.cancel();
      _state = GemmaModelState.ready;
      _progress = 1.0;
      notifyListeners();
      debugPrint('[Gemma] ✅ Download complete in $_elapsed');
    } catch (e) {
      _state = GemmaModelState.error;
      _error = e.toString();
      notifyListeners();
      debugPrint('[Gemma] ❌ Download failed after $_elapsed: $e');
    } finally {
      heartbeat?.cancel();
    }
  }

  void _ensureLifecycleObserver() {
    if (_observingLifecycle) return;
    WidgetsBinding.instance.addObserver(this);
    _observingLifecycle = true;
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state != AppLifecycleState.resumed) return;
    if (_state != GemmaModelState.error) return;
    unawaited(_resumeDownloadOnForeground());
  }

  Future<void> _resumeDownloadOnForeground() async {
    if (_resumeOnForegroundRunning) return;
    _resumeOnForegroundRunning = true;
    try {
      debugPrint('[Gemma] App returned to foreground — resuming download');
      await retry();
    } finally {
      _resumeOnForegroundRunning = false;
    }
  }

  void _maybeLogProgress(double p) {
    final pct = (p * 100).truncate();
    // Log at every 5% milestone.
    final milestone = (pct / 5).floor() * 5;
    if (milestone > (_lastLoggedProgress * 100).truncate()) {
      _lastLoggedProgress = p;
      debugPrint('[Gemma] 📥 $pct% downloaded ($_elapsed)');
    }
  }

  /// Retry after failure.
  Future<void> retry() async {
    debugPrint('[Gemma] Retrying download…');
    _started = false;
    _error = null;
    _progress = 0.0;
    _downloadStartedAt = null;
    _state = GemmaModelState.notDownloaded;
    notifyListeners();
    await ensureStarted();
  }
}
