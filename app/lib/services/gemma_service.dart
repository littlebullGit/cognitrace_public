import 'dart:async';
import 'dart:convert';
import 'dart:developer' as developer;
import 'dart:io';
import 'dart:math' show max, min;

import 'package:flutter/foundation.dart';
import 'package:llamadart/llamadart.dart';
import 'package:path_provider/path_provider.dart';

import '../l10n/app_strings.dart';
import '../models/doctor_discussion_guide.dart';
import '../models/knowledge_card.dart';
import 'language_preference_service.dart';
import 'knowledge_base_service.dart';
import 'voice_metric_policy.dart';

enum GemmaModelState { notDownloaded, downloading, ready, loading, error }

enum _FollowUpIntent {
  resultMeaning,
  featureMeaning,
  nextSteps,
  confounders,
  limitations,
  general,
}

/// Topics trackable during a practice conversation (v1: intentionally simple).
enum RehearsalTopic { visitReason, screeningResult, nextSteps }

class _GemmaSessionState {
  const _GemmaSessionState({
    required this.language,
    required this.riskLevel,
    required this.riskScore,
    required this.allFeatures,
    required this.primaryFindings,
    required this.modelScores,
  });

  final String language;
  final String riskLevel;
  final double riskScore;
  final Map<String, double> allFeatures;
  final List<String> primaryFindings;
  final Map<String, double> modelScores;
}

class _RehearsalState {
  _RehearsalState({required this.guide, required this.language});
  final DoctorDiscussionGuide guide;
  final String language;
  final List<({String role, String text})> history = [];
  final Set<RehearsalTopic> topicsCovered = {};
}

/// Gemma 4 E2B GGUF (Q4_K_M, about 2.7 GB) via llamadart / llama.cpp.
///
/// - Apache 2.0, no auth required
/// - iOS Metal GPU acceleration (automatic)
/// - Chat template auto-detected from GGUF metadata
const _gemmaModelUrl =
    'https://huggingface.co/littlebull9/cognitrace-gemma4-medical-GGUF/resolve/main/cognitrace-gemma4-medical-v3-Q4_K_M.gguf';

const _gemmaModelFilename = 'cognitrace-gemma4-medical-v3-Q4_K_M.gguf';

/// Compressed clinical reference ranges (Praat-based, peer-reviewed).
///
/// Embedded in the system prompt so Gemma can interpret biomarker values
/// against published healthy/PD baselines during both initial analysis
/// and follow-up chat questions.
///
/// Sources: Little 2009 (IEEE TBME), Tsanas 2012 (IEEE TBME),
/// Favaro 2023 (Front. Neurol.), Valzania 2021 (Front. Neurol.),
/// npj Health Systems 2024.
const _clinicalKnowledge = '''
ABOUT THIS SCREENING:
CogniTrace extracts 56 acoustic biomarkers from a 60-second voice recording. The shipped risk score is produced from the full 56-marker voice profile. Gemma does not produce the risk score. It turns the result into plain-language guidance and grounded follow-up support.

The test measures: voice steadiness (pitch control and regularity), vocal clarity (tone quality and harmonics), speech energy (volume and power), and spectral patterns (frequency distribution). In Parkinson's disease, these change due to effects on vocal fold control, breathing, and articulation — often years before visible motor symptoms.

IMPORTANT RULES FOR INTERPRETING RESULTS:
- The risk score IS the result. It reflects the full 56-marker voice profile.
- A few perturbation metrics may include [ABOVE REF] or [WITHIN REF] flags against classic sustained-vowel reference values. These are soft reference flags, not universal normal/abnormal labels.
- Features WITHOUT reference flags should be discussed cautiously and in context.
- No single biomarker should be described as the reason for the classifier result.
- The models were trained on a clinical dataset and already account for normal variation.''';

class GemmaService {
  static GemmaModelState _state = GemmaModelState.notDownloaded;
  static String? _lastError;
  static LlamaEngine? _engine;
  static String? _modelPath;
  static _GemmaSessionState? _lastSessionState;
  static final Map<String, Map<String, KnowledgeCard>> _localizedCardCache = {};
  static final Map<String, String> _educationIntroCache = {};
  static _RehearsalState? _rehearsalState;
  static const _maxRehearsalHistoryTurns = 4;
  static const _softTurnLimit = 6;

  static GemmaModelState get state => _state;
  static String? get lastError => _lastError;
  static String get modelUrl => _gemmaModelUrl;
  static String get modelFilename => _gemmaModelFilename;

  /// Whether follow-up chat has the compact screening state it needs.
  static bool get hasActiveSession =>
      _engine != null && _lastSessionState != null;

  /// Check if the GGUF model file exists on disk.
  static Future<void> initialize() async {
    final dir = await getApplicationDocumentsDirectory();
    _modelPath = '${dir.path}/$_gemmaModelFilename';

    if (await File(_modelPath!).exists()) {
      final size = await File(_modelPath!).length();
      debugPrint('[Gemma] Model found: $_modelPath ($size bytes)');
      _state = GemmaModelState.ready;
    } else {
      debugPrint('[Gemma] Model not found at $_modelPath');
    }
  }

  /// Download the GGUF model file from HuggingFace.
  static Future<void> downloadModel({
    void Function(double progress)? onProgress,
  }) async {
    if (_state == GemmaModelState.ready) return;
    _state = GemmaModelState.downloading;
    _lastError = null;

    final dir = await getApplicationDocumentsDirectory();
    _modelPath = '${dir.path}/$_gemmaModelFilename';
    final tmpPath = '$_modelPath.tmp';
    final tmpFile = File(tmpPath);
    final client = HttpClient();

    try {
      var resumeFrom = await tmpFile.exists() ? await tmpFile.length() : 0;
      debugPrint('[Gemma] Downloading $_gemmaModelUrl');
      if (resumeFrom > 0) {
        debugPrint('[Gemma] Resuming partial download at $resumeFrom bytes');
      }

      var response = await _openModelDownload(client, resumeFrom: resumeFrom);
      if (response.statusCode == HttpStatus.requestedRangeNotSatisfiable &&
          resumeFrom > 0) {
        await response.drain<void>();
        debugPrint('[Gemma] Saved partial download is stale; restarting');
        await tmpFile.delete();
        resumeFrom = 0;
        response = await _openModelDownload(client, resumeFrom: 0);
      }

      if (response.statusCode != HttpStatus.ok &&
          response.statusCode != HttpStatus.partialContent) {
        throw Exception('HTTP ${response.statusCode}');
      }

      final contentRange = response.headers.value(
        HttpHeaders.contentRangeHeader,
      );
      final responseStart = _contentRangeStart(contentRange);
      var appending =
          resumeFrom > 0 &&
          response.statusCode == HttpStatus.partialContent &&
          responseStart == resumeFrom;

      if (resumeFrom > 0 && response.statusCode == HttpStatus.ok) {
        debugPrint('[Gemma] Server ignored range request; restarting');
      } else if (resumeFrom > 0 && !appending) {
        await response.drain<void>();
        debugPrint(
          '[Gemma] Range response did not match partial file; restarting',
        );
        await tmpFile.delete();
        resumeFrom = 0;
        response = await _openModelDownload(client, resumeFrom: 0);
        if (response.statusCode != HttpStatus.ok) {
          throw Exception('HTTP ${response.statusCode}');
        }
        appending = false;
      }

      final totalBytes = appending
          ? _contentRangeTotal(contentRange) ??
                (response.contentLength > 0
                    ? resumeFrom + response.contentLength
                    : null)
          : response.contentLength > 0
          ? response.contentLength
          : null;

      debugPrint(
        '[Gemma] Content-Length: ${response.contentLength} bytes'
        '${totalBytes == null ? '' : ' (total $totalBytes bytes)'}',
      );

      final sink = tmpFile.openWrite(
        mode: appending ? FileMode.append : FileMode.write,
      );
      var received = appending ? resumeFrom : 0;
      if (totalBytes != null && totalBytes > 0) {
        onProgress?.call((received / totalBytes).clamp(0.0, 1.0));
      }

      try {
        await for (final chunk in response) {
          sink.add(chunk);
          received += chunk.length;
          if (totalBytes != null && totalBytes > 0) {
            onProgress?.call((received / totalBytes).clamp(0.0, 1.0));
          }
        }
      } finally {
        await sink.flush();
        await sink.close();
      }

      if (totalBytes != null && received < totalBytes) {
        throw Exception(
          'Download interrupted at $received of $totalBytes bytes',
        );
      }

      // Rename tmp -> final (atomic on same filesystem).
      await tmpFile.rename(_modelPath!);

      final finalSize = await File(_modelPath!).length();
      debugPrint('[Gemma] Download complete: $finalSize bytes');
      _state = GemmaModelState.ready;
    } catch (e) {
      debugPrint('[Gemma] Keeping partial download for retry: $tmpPath');
      _state = GemmaModelState.error;
      _lastError = e.toString();
      rethrow;
    } finally {
      client.close(force: true);
    }
  }

  static Future<HttpClientResponse> _openModelDownload(
    HttpClient client, {
    required int resumeFrom,
  }) async {
    final request = await client.getUrl(Uri.parse(_gemmaModelUrl));
    if (resumeFrom > 0) {
      request.headers.set(HttpHeaders.rangeHeader, 'bytes=$resumeFrom-');
    }
    return request.close();
  }

  static int? _contentRangeStart(String? value) {
    final match = _contentRangePattern.firstMatch(value ?? '');
    if (match == null) return null;
    return int.tryParse(match.group(1)!);
  }

  static int? _contentRangeTotal(String? value) {
    final match = _contentRangePattern.firstMatch(value ?? '');
    if (match == null) return null;
    final total = match.group(3);
    if (total == null || total == '*') return null;
    return int.tryParse(total);
  }

  static final _contentRangePattern = RegExp(r'^bytes (\d+)-(\d+)/(\d+|\*)$');

  /// Log via dart:developer so lines reach Xcode Console / idevicesyslog
  /// even in release builds (where debugPrint is stripped).
  static void _log(String message, {Object? error, StackTrace? stackTrace}) {
    developer.log(message, name: 'Gemma', error: error, stackTrace: stackTrace);
    debugPrint('[Gemma] $message');
  }

  /// Load the model into the llama.cpp engine.
  ///
  /// Concurrent callers (e.g. language switch firing a second analyze while
  /// the first is still loading) are coalesced onto the same [_loadingFuture].
  /// Without this, the second caller saw `_engine != null` after the engine
  /// handle was constructed but before `LlamaEngine.loadModel` finished, and
  /// tripped `Engine not ready` on chat creation.
  ///
  /// The inner `_loadModelAttempt` still does one internal retry on actual
  /// native-init failures, but the common first-run error we were seeing was
  /// the concurrency race, not a genuine init failure.
  static Future<void>? _loadingFuture;

  static Future<void> loadModel() async {
    if (_state == GemmaModelState.ready && _engine != null) return;
    if (_loadingFuture != null) return _loadingFuture!;
    if (_modelPath == null) await initialize();
    if (_state != GemmaModelState.ready) {
      throw StateError('Model not downloaded');
    }
    final future = _loadModelAttempt(attempt: 1);
    _loadingFuture = future.whenComplete(() => _loadingFuture = null);
    return _loadingFuture!;
  }

  static const _loadModelMaxAttempts = 2;
  static const _loadModelRetryDelay = Duration(seconds: 3);

  static Future<void> _loadModelAttempt({required int attempt}) async {
    _state = GemmaModelState.loading;
    _log('Loading model into llama.cpp engine (attempt $attempt)\u2026');
    final sw = Stopwatch()..start();

    try {
      _engine = LlamaEngine(LlamaBackend());
      await _engine!.loadModel(
        _modelPath!,
        modelParams: ModelParams(
          contextSize: 2048, // Expanded from 1024 for clinical context + chat
          batchSize: 512,
          microBatchSize:
              256, // Critical: Gemma head_size=256 causes huge buffers
          gpuLayers: 999, // All 35 layers on Metal GPU
          preferredBackend: GpuBackend.metal,
        ),
      );
      _state = GemmaModelState.ready;
      _log('Model loaded in ${sw.elapsedMilliseconds}ms');
    } catch (e, st) {
      _log(
        'loadModel attempt $attempt failed after ${sw.elapsedMilliseconds}ms',
        error: e,
        stackTrace: st,
      );
      await _engine?.dispose();
      _engine = null;

      if (attempt < _loadModelMaxAttempts) {
        _log(
          'Retrying loadModel in ${_loadModelRetryDelay.inSeconds}s '
          '(Metal/ONNX warmup contention)',
        );
        await Future<void>.delayed(_loadModelRetryDelay);
        await _loadModelAttempt(attempt: attempt + 1);
        return;
      }

      _state = GemmaModelState.error;
      _lastError = e.toString();
      rethrow;
    }
  }

  static const _inferenceTimeout = Duration(minutes: 3);

  static const _defaultGenParams = GenerationParams(
    maxTokens: 512,
    temp: 0.7,
    topK: 40,
  );

  /// Shorter limit for the initial 2-sentence summary.
  static const _summaryGenParams = GenerationParams(
    maxTokens: 150,
    temp: 0.7,
    topK: 40,
  );

  /// Generate the initial clinical narrative from voice biomarkers.
  ///
  /// The [ChatSession] is kept alive after this call so users can ask
  /// follow-up questions via [askFollowUp] / [askFollowUpStream].
  static Future<String> analyze({
    required Map<String, double> features,
    required double riskScore,
    required String language,
    Map<String, double>? modelScores,
  }) async {
    await loadModel(); // idempotent + coalesces concurrent callers

    final riskLevel = riskScore >= 0.7
        ? 'elevated'
        : riskScore >= 0.4
        ? 'moderate'
        : 'low';

    // Retain a compact state for bounded follow-up chat.
    _lastSessionState = _buildSessionState(
      features: features,
      riskScore: riskScore,
      riskLevel: riskLevel,
      language: language,
      modelScores: modelScores ?? const {},
    );

    final systemPrompt =
        'You are CogniTrace, an on-device clinical voice analysis assistant '
        'powered by Gemma. You interpret acoustic biomarker results from voice '
        'recordings, explaining what they mean for neurological health.\n\n'
        '$_clinicalKnowledge\n'
        'GUIDELINES:\n'
        '- Respond in $language\n'
        '- This is a SCREENING tool, never a diagnosis\n'
        '- Be direct and professional. Not overly warm or overly alarming.\n'
        '- NEVER use markdown in the initial summary (plain text only)\n'
        '- For follow-up answers, USE markdown formatting for readability: '
        '## headers for sections, **bold** for feature names, bullet points '
        'for lists, blank lines between paragraphs\n'
        '- NEVER repeat a phrase you already said. Every sentence must add '
        'new information.\n'
        '- Initial summary: EXACTLY 2 sentences. Use simple everyday words.\n'
        '- Examples of good summaries:\n'
        '  * Low: "Your voice patterns are within normal range. '
        'No concerning voice patterns found."\n'
        '  * Moderate: "Some voice patterns show minor changes. '
        'Consider a follow-up check in a few months."\n'
        '  * Elevated: "Several voice patterns suggest changes that '
        'should be evaluated by a doctor. Please schedule a visit '
        'with your healthcare provider."\n'
        '- Be factual. Do NOT add unnecessary positivity or softening '
        'for moderate or elevated results.\n'
        '- Do NOT list features or mention biomarker names in the summary.\n'
        '- The summary is risk-based only. Do not imply model attribution.\n'
        '- For low risk: purely reassuring. Do NOT mention anything outside '
        'range, suggest seeing a doctor, or recommend regular monitoring/checks. '
        'Just say the voice sounds great.\n'
        '- For moderate risk: note gently, suggest periodic monitoring\n'
        '- For elevated risk: encourage healthcare provider visit\n'
        '- Never say "diagnosis", "diagnosed", or "disease"\n'
        '- Only discuss voice health and screening results. '
        'Politely redirect unrelated topics.';

    final modelLine = modelScores != null && modelScores.isNotEmpty
        ? 'Model agreement: ${modelScores.entries.map((e) => '${e.key}=${e.value.toStringAsFixed(3)}').join(', ')}\n'
        : '';

    final userPrompt =
        'Voice screening result:\n'
        'Overall: $riskLevel risk (score: ${riskScore.toStringAsFixed(2)})\n'
        '$modelLine\n'
        'Write exactly 2 sentences matching the $riskLevel risk examples above. '
        'Do not soften or contradict the risk level. '
        'Do not mention biomarker names or explain the model. '
        'Do not use medical jargon. Respond in $language.';

    final sw = Stopwatch()..start();
    debugPrint('[Gemma] Creating chat session\u2026');

    final chat = ChatSession(_engine!, systemPrompt: systemPrompt);
    debugPrint('[Gemma] Sending query (${userPrompt.length} chars)\u2026');

    final buffer = StringBuffer();
    try {
      await for (final chunk
          in chat
              .create(
                [LlamaTextContent(userPrompt)],
                enableThinking: false,
                params: _summaryGenParams,
              )
              .timeout(_inferenceTimeout)) {
        final content = chunk.choices.firstOrNull?.delta.content;
        if (content != null) buffer.write(content);
      }
    } on TimeoutException {
      debugPrint(
        '[Gemma] \u274c Inference timed out after ${sw.elapsedMilliseconds}ms',
      );
      rethrow;
    }

    sw.stop();
    final response = buffer.toString().trim();
    debugPrint(
      '[Gemma] \u2705 Response (${response.length} chars) in ${sw.elapsedMilliseconds}ms',
    );
    return response;
  }

  static String _followUpSystemPrompt(String language) =>
      'LANGUAGE: You MUST respond in $language. Every word of your reply must be in $language.\n\n'
      'You are CogniTrace, an on-device clinical voice screening explainer.\n\n'
      'Rules:\n'
      '- This is a SCREENING tool, never a diagnosis\n'
      '- The classifier already produced the risk score; do not override it\n'
      '- Use only the CURRENT SCREENING STATE and REFERENCE CARDS provided\n'
      '- If the user asks beyond those materials, say the app cannot answer that reliably\n'
      '- Keep answers concise and specific to the current result\n'
      '- Use markdown for follow-up answers when it helps readability\n'
      '- When needed for safe interpretation, say the result comes from the full 56-marker voice profile\n'
      '- Do not claim that one biomarker caused the screening result\n'
      '- You may explain what a biomarker measures in general terms\n'
      '- Never give treatment advice or claim certainty\n'
      '- Always respond in $language, regardless of what language the instructions are in';

  static Future<String> _buildFollowUpPrompt(
    _GemmaSessionState state,
    String question,
  ) async {
    final intent = _classifyFollowUpIntent(question);
    final focusedFeatures = _extractFeatureFocus(question);
    final featureLines = _selectFeatureLines(
      state: state,
      intent: intent,
      focusedFeatures: focusedFeatures,
    );
    final personalizedInsights = _buildPersonalizedInsights(
      state: state,
      intent: intent,
      focusedFeatures: focusedFeatures,
    );
    final cards = await _selectKnowledgeCards(
      state: state,
      intent: intent,
      focusedFeatures: focusedFeatures,
    );

    final buffer = StringBuffer()
      ..writeln('CURRENT SCREENING STATE:')
      ..writeln(
        '- risk: ${state.riskLevel} (${state.riskScore.toStringAsFixed(2)})',
      )
      ..writeln('- screening basis: 56 acoustic voice markers')
      ..writeln('- marker table: 56 values available for drill-down')
      ..writeln('- model internals are not user-facing');

    if (featureLines.isNotEmpty) {
      buffer.writeln();
      buffer.writeln('RELEVANT BIOMARKERS:');
      for (final line in featureLines) {
        buffer.writeln('- $line');
      }
    }

    if (personalizedInsights.isNotEmpty) {
      buffer.writeln();
      buffer.writeln('PERSONALIZED INTERPRETATION HINTS:');
      for (final insight in personalizedInsights) {
        buffer.writeln('- $insight');
      }
    }

    if (cards.isNotEmpty) {
      buffer.writeln();
      buffer.writeln('REFERENCE CARDS:');
      for (final card in cards) {
        buffer.writeln('- ${card.id}: ${card.title} — ${card.body}');
      }
    }

    buffer
      ..writeln()
      ..writeln('QUESTION:')
      ..writeln(question)
      ..writeln()
      ..writeln('RESPONSE INSTRUCTIONS:')
      ..writeln('- Answer using the screening state and reference cards')
      ..writeln(
        '- When discussing any feature, cite the user value and explain it in cautious, non-causal terms',
      )
      ..writeln(
        '- Do not claim to know which single feature or feature combination caused the classifier result',
      )
      ..writeln('- Do not restate unrelated biomarker details')
      ..writeln(
        '- If a biomarker is not listed here, say the app tracks more detail in the biomarker table',
      )
      ..writeln(
        '- Prefer personalized interpretation over generic textbook wording',
      )
      ..writeln(
        '- Be careful: one biomarker alone does not determine the result',
      )
      ..writeln(
        '- Keep the answer grounded in the current result and within screening scope',
      );

    return buffer.toString();
  }

  /// Send a follow-up question using a fresh, compact prompt per turn.
  static Future<String> askFollowUp(String question) async {
    if (_engine == null || _lastSessionState == null) {
      throw StateError('No active chat session. Call analyze() first.');
    }

    final state = _lastSessionState!;
    final prompt = await _buildFollowUpPrompt(state, question);
    debugPrint(
      '[Gemma] Follow-up: "${question.substring(0, question.length.clamp(0, 50))}\u2026"',
    );
    final sw = Stopwatch()..start();
    final buffer = StringBuffer();
    final chat = ChatSession(
      _engine!,
      systemPrompt: _followUpSystemPrompt(state.language),
    );

    try {
      await for (final chunk
          in chat
              .create(
                [LlamaTextContent(prompt)],
                enableThinking: false,
                params: _defaultGenParams,
              )
              .timeout(_inferenceTimeout)) {
        final content = chunk.choices.firstOrNull?.delta.content;
        if (content != null) buffer.write(content);
      }
    } on TimeoutException {
      debugPrint(
        '[Gemma] \u274c Follow-up timed out after ${sw.elapsedMilliseconds}ms',
      );
      rethrow;
    }

    sw.stop();
    final response = buffer.toString().trim();
    debugPrint(
      '[Gemma] \u2705 Follow-up (${response.length} chars) in ${sw.elapsedMilliseconds}ms',
    );
    return response;
  }

  /// Stream a follow-up response token-by-token for live chat display.
  static Stream<String> askFollowUpStream(String question) async* {
    if (_engine == null || _lastSessionState == null) {
      throw StateError('No active chat session. Call analyze() first.');
    }

    final state = _lastSessionState!;
    final prompt = await _buildFollowUpPrompt(state, question);
    final chat = ChatSession(
      _engine!,
      systemPrompt: _followUpSystemPrompt(state.language),
    );

    debugPrint(
      '[Gemma] Streaming follow-up: "${question.substring(0, question.length.clamp(0, 50))}\u2026"',
    );

    await for (final chunk
        in chat
            .create(
              [LlamaTextContent(prompt)],
              enableThinking: false,
              params: _defaultGenParams,
            )
            .timeout(_inferenceTimeout)) {
      final content = chunk.choices.firstOrNull?.delta.content;
      if (content != null) yield content;
    }
  }

  static Future<String> switchLanguage({
    required Map<String, double> features,
    required double riskScore,
    required String newLanguage,
    Map<String, double>? modelScores,
  }) async {
    return analyze(
      features: features,
      riskScore: riskScore,
      language: newLanguage,
      modelScores: modelScores,
    );
  }

  /// Remove the model file from disk to force re-download.
  static Future<void> uninstallModel() async {
    await dispose();
    if (_modelPath != null) {
      try {
        await File(_modelPath!).delete();
        debugPrint('[Gemma] Model file deleted');
      } catch (_) {}
    }
    _state = GemmaModelState.notDownloaded;
  }

  static Future<void> dispose() async {
    _lastSessionState = null;
    _rehearsalState = null;
    _localizedCardCache.clear();
    _educationIntroCache.clear();
    await _engine?.dispose();
    _engine = null;
  }

  static Future<String> generateEducationIntro({
    required double riskScore,
    required String language,
    Map<String, double> modelScores = const {},
  }) async {
    await loadModel(); // idempotent + coalesces concurrent callers

    final riskLevel = riskScore >= 0.7
        ? 'elevated'
        : riskScore >= 0.4
        ? 'moderate'
        : 'low';
    final agreementBucket = _modelAgreementBucket(modelScores);
    final cacheKey =
        '$language|$riskLevel|${riskScore.toStringAsFixed(2)}|$agreementBucket';
    final cached = _educationIntroCache[cacheKey];
    if (cached != null) return cached;

    final gemmaLanguage = LanguagePreferenceService.gemmaNameFor(language);
    final systemPrompt =
        'You write a short note that appears at the top of the "Understand result" view in a Parkinson voice screening app.\n'
        'Rules:\n'
        '- Respond in $gemmaLanguage\n'
        '- This is a SCREENING tool, never a diagnosis\n'
        '- Keep it to 1 or 2 short sentences\n'
        '- Do not mention biomarker names\n'
        '- Do not explain the model internals\n'
        '- Do not claim certainty\n'
        '- Do not use markdown, bullets, headings, or emphasis markers\n'
        '- The detailed educational cards below are already localized and reviewed, so do not repeat their wording\n'
        '- Add only the most useful practical takeaway for this user right now\n'
        '- For low risk: reassuring, educational, baseline-oriented\n'
        '- For moderate risk: cautious, contextual, mentions monitoring and conditions\n'
        '- For elevated risk: calm, practical, follow-up oriented\n';

    final userPrompt =
        'Current result:\n'
        '- risk: $riskLevel (${riskScore.toStringAsFixed(2)})\n'
        '- screening basis: 56 acoustic voice markers\n'
        'Write a short note for the education sheet explaining what to keep in mind right now. '
        'This note should complement the static educational cards below it, not duplicate them.';

    final chat = ChatSession(_engine!, systemPrompt: systemPrompt);
    final buffer = StringBuffer();
    await for (final chunk
        in chat
            .create(
              [LlamaTextContent(userPrompt)],
              enableThinking: false,
              params: const GenerationParams(
                maxTokens: 96,
                temp: 0.2,
                topK: 40,
              ),
            )
            .timeout(_inferenceTimeout)) {
      final content = chunk.choices.firstOrNull?.delta.content;
      if (content != null) buffer.write(content);
    }

    final intro = buffer.toString().trim();
    if (intro.isEmpty) {
      return _fallbackEducationIntro(riskLevel: riskLevel);
    }
    final sanitized = _sanitizePlainText(intro);
    _educationIntroCache[cacheKey] = sanitized;
    return sanitized;
  }

  static String _modelAgreementBucket(Map<String, double> modelScores) {
    if (modelScores.length < 2) return 'unknown';
    final values = modelScores.values.toList();
    final spread = values.reduce(max) - values.reduce(min);
    if (spread < 0.08) return 'fairly consistent';
    if (spread < 0.18) return 'somewhat mixed';
    return 'mixed';
  }

  static String _fallbackEducationIntro({required String riskLevel}) {
    return switch (riskLevel) {
      'low' =>
        'This result is best read as reassuring under these recording conditions. It can also serve as a useful baseline if you decide to check again later.',
      'moderate' =>
        'This result is not definitive on its own. It is best understood together with recording conditions, symptoms, and whether the pattern changes over time.',
      _ =>
        'This result is worth taking seriously, but it is still not a diagnosis. The most useful next step is to understand the result clearly and bring it into a clinician conversation.',
    };
  }

  static Future<List<KnowledgeCard>> localizeKnowledgeCards({
    required List<KnowledgeCard> cards,
    required String language,
  }) async {
    if (language == 'English' || cards.isEmpty) return cards;
    await loadModel(); // idempotent + coalesces concurrent callers

    final cacheForLanguage = _localizedCardCache.putIfAbsent(
      language,
      () => {},
    );
    final missing = <KnowledgeCard>[];
    final ordered = <KnowledgeCard>[];
    for (final card in cards) {
      final cached = cacheForLanguage[card.id];
      if (cached != null) {
        ordered.add(cached);
      } else {
        missing.add(card);
      }
    }

    if (missing.isNotEmpty) {
      final gemmaLanguage = LanguagePreferenceService.gemmaNameFor(language);
      final payload = jsonEncode(
        missing
            .map(
              (card) => {'id': card.id, 'title': card.title, 'body': card.body},
            )
            .toList(),
      );
      final systemPrompt =
          'You translate educational reference cards for an on-device health screening app.\n'
          'Rules:\n'
          '- Translate into $gemmaLanguage\n'
          '- Preserve the card id exactly\n'
          '- Translate only title and body\n'
          '- Keep the meaning cautious and patient-safe\n'
          '- Do not add diagnosis or treatment language\n'
          '- Do not use markdown, bullets, headings, or emphasis markers\n'
          '- Return valid JSON only as an array of objects with id, title, body\n';
      final userPrompt =
          'Translate these cards into $gemmaLanguage.\n'
          'Return JSON only.\n'
          '$payload';

      final chat = ChatSession(_engine!, systemPrompt: systemPrompt);
      final buffer = StringBuffer();
      await for (final chunk
          in chat
              .create(
                [LlamaTextContent(userPrompt)],
                enableThinking: false,
                params: const GenerationParams(
                  maxTokens: 1200,
                  temp: 0.2,
                  topK: 40,
                ),
              )
              .timeout(_inferenceTimeout)) {
        final content = chunk.choices.firstOrNull?.delta.content;
        if (content != null) buffer.write(content);
      }
      final localized = _parseCardTranslations(buffer.toString());
      for (final card in missing) {
        final translated = localized[card.id];
        if (translated != null) {
          cacheForLanguage[card.id] = card.copyWith(
            title: _sanitizePlainText(translated['title'] ?? ''),
            body: _sanitizePlainText(translated['body'] ?? ''),
          );
        }
      }
    }

    return cards
        .map((card) => cacheForLanguage[card.id] ?? card)
        .toList(growable: false);
  }

  static Map<String, Map<String, String>> _parseCardTranslations(
    String response,
  ) {
    final start = response.indexOf('[');
    if (start < 0) return const {};
    final end = response.lastIndexOf(']');
    if (end <= start) return const {};
    try {
      final parsed =
          jsonDecode(response.substring(start, end + 1)) as List<dynamic>;
      final result = <String, Map<String, String>>{};
      for (final entry in parsed) {
        final map = entry as Map<String, dynamic>;
        final id = map['id']?.toString();
        final title = map['title']?.toString();
        final body = map['body']?.toString();
        if (id == null || title == null || body == null) continue;
        result[id] = {'title': title, 'body': body};
      }
      return result;
    } catch (_) {
      return const {};
    }
  }

  static String _sanitizePlainText(String text) {
    var result = text.trim();
    result = result
        .replaceAll('**', '')
        .replaceAll('__', '')
        .replaceAll('`', '')
        .replaceAll('###', '')
        .replaceAll('##', '')
        .replaceAll('#', '');
    result = result.replaceAllMapped(
      RegExp(r'^\s*[-*>]+\s?', multiLine: true),
      (match) => '',
    );
    result = result.replaceAll(RegExp(r'\n{3,}'), '\n\n');
    return result.trim();
  }

  @visibleForTesting
  static Future<String> debugBuildFollowUpPrompt({
    required Map<String, double> features,
    required double riskScore,
    required String language,
    required String question,
    Map<String, double> modelScores = const {},
  }) async {
    final riskLevel = riskScore >= 0.7
        ? 'elevated'
        : riskScore >= 0.4
        ? 'moderate'
        : 'low';
    final state = _buildSessionState(
      features: features,
      riskScore: riskScore,
      riskLevel: riskLevel,
      language: language,
      modelScores: modelScores,
    );
    return _buildFollowUpPrompt(state, question);
  }

  @visibleForTesting
  static Future<List<String>> debugKnowledgeCardIds({
    required Map<String, double> features,
    required double riskScore,
    required String language,
    required String question,
  }) async {
    final riskLevel = riskScore >= 0.7
        ? 'elevated'
        : riskScore >= 0.4
        ? 'moderate'
        : 'low';
    final state = _buildSessionState(
      features: features,
      riskScore: riskScore,
      riskLevel: riskLevel,
      language: language,
      modelScores: const {},
    );
    final intent = _classifyFollowUpIntent(question);
    final focus = _extractFeatureFocus(question);
    return (await _selectKnowledgeCards(
      state: state,
      intent: intent,
      focusedFeatures: focus,
    )).map((card) => card.id).toList();
  }

  @visibleForTesting
  static Future<List<String>> debugDoctorGuideKnowledgeCardIds({
    required Map<String, double> features,
    required double riskScore,
    required String language,
    String? userNotes,
    Map<String, double> modelScores = const {},
  }) async {
    final riskLevel = riskScore >= 0.7
        ? 'elevated'
        : riskScore >= 0.4
        ? 'moderate'
        : 'low';
    final state = _buildSessionState(
      features: features,
      riskScore: riskScore,
      riskLevel: riskLevel,
      language: language,
      modelScores: modelScores,
    );
    return (await _selectDoctorGuideCards(
      state: state,
      userContext: _parseDoctorGuideUserContext(userNotes),
    )).map((card) => card.id).toList();
  }

  static List<String> suggestedQuestions(String language) {
    final state = _lastSessionState;
    if (state == null) return const [];

    final keyFeature = _topSuggestedFeature(state);
    final featureLabel = _localizedFeatureLabel(keyFeature, language);
    final prompts = state.riskLevel == 'low'
        ? <String>[
            _localizedResultQuestion(state.riskLevel, language),
            _localizedVoiceWhyQuestion(language),
            _localizedRepeatLaterQuestion(language),
            _localizedLimitsQuestion(language),
            _localizedFeatureQuestion(featureLabel, language),
          ]
        : <String>[
            _localizedResultQuestion(state.riskLevel, language),
            _localizedConfounderQuestion(language),
            _localizedLimitsQuestion(language),
            _localizedDoctorPrepQuestion(language),
            _localizedFeatureQuestion(featureLabel, language),
          ];

    final unique = <String>[];
    for (final prompt in prompts) {
      if (!unique.contains(prompt)) unique.add(prompt);
    }
    return unique;
  }

  @visibleForTesting
  static List<String> debugSuggestedQuestions({
    required Map<String, double> features,
    required double riskScore,
    required String language,
    Map<String, double> modelScores = const {},
  }) {
    final riskLevel = riskScore >= 0.7
        ? 'elevated'
        : riskScore >= 0.4
        ? 'moderate'
        : 'low';
    _lastSessionState = _buildSessionState(
      features: features,
      riskScore: riskScore,
      riskLevel: riskLevel,
      language: language,
      modelScores: modelScores,
    );
    return suggestedQuestions(language);
  }

  static const _featureLabels = <String, String>{
    'f0_mean': 'average pitch',
    'f0_std': 'pitch variability',
    'voiced_fraction': 'voiced fraction',
    'jitter_local': 'jitter',
    'jitter_rap': 'jitter RAP',
    'jitter_ppq5': 'jitter PPQ5',
    'shimmer_local': 'shimmer',
    'shimmer_apq3': 'shimmer APQ3',
    'hnr': 'harmonics-to-noise ratio',
    'rms_energy': 'voice energy',
    'log_energy': 'log energy',
    'spec_centroid_mean': 'spectral centroid',
    'spec_bandwidth_mean': 'spectral bandwidth',
  };

  static const _localizedFeatureLabels = <String, Map<String, String>>{
    'English': {
      'f0_mean': 'average pitch',
      'f0_std': 'pitch variation',
      'voiced_fraction': 'voiced fraction',
      'jitter_local': 'jitter',
      'jitter_rap': 'jitter RAP',
      'jitter_ppq5': 'jitter PPQ5',
      'shimmer_local': 'shimmer',
      'shimmer_apq3': 'shimmer APQ3',
      'hnr': 'harmonics-to-noise ratio',
      'rms_energy': 'voice energy',
      'log_energy': 'log energy',
      'spec_centroid_mean': 'spectral centroid',
      'spec_bandwidth_mean': 'spectral bandwidth',
    },
    'Italiano': {
      'f0_mean': 'altezza media',
      'f0_std': 'variazione del tono',
      'voiced_fraction': 'frazione sonora',
      'jitter_local': 'jitter',
      'jitter_rap': 'jitter RAP',
      'jitter_ppq5': 'jitter PPQ5',
      'shimmer_local': 'shimmer',
      'shimmer_apq3': 'shimmer APQ3',
      'hnr': 'rapporto armoniche-rumore',
      'rms_energy': 'energia vocale',
      'log_energy': 'energia logaritmica',
      'spec_centroid_mean': 'centroide spettrale',
      'spec_bandwidth_mean': 'larghezza di banda spettrale',
    },
    '中文': {
      'f0_mean': '平均基频',
      'f0_std': '音高波动',
      'voiced_fraction': '有声音帧比例',
      'jitter_local': '抖动',
      'jitter_rap': '抖动 RAP',
      'jitter_ppq5': '抖动 PPQ5',
      'shimmer_local': '闪烁',
      'shimmer_apq3': '闪烁 APQ3',
      'hnr': '谐噪比',
      'rms_energy': '声音能量',
      'log_energy': '对数能量',
      'spec_centroid_mean': '频谱质心',
      'spec_bandwidth_mean': '频谱带宽',
    },
    'Español': {
      'f0_mean': 'tono medio',
      'f0_std': 'variación del tono',
      'voiced_fraction': 'fracción sonora',
      'jitter_local': 'jitter',
      'jitter_rap': 'jitter RAP',
      'jitter_ppq5': 'jitter PPQ5',
      'shimmer_local': 'shimmer',
      'shimmer_apq3': 'shimmer APQ3',
      'hnr': 'relación armónicos-ruido',
      'rms_energy': 'energía vocal',
      'log_energy': 'energía logarítmica',
      'spec_centroid_mean': 'centroide espectral',
      'spec_bandwidth_mean': 'ancho de banda espectral',
    },
    'Français': {
      'f0_mean': 'hauteur moyenne',
      'f0_std': 'variation de la hauteur',
      'voiced_fraction': 'fraction voisée',
      'jitter_local': 'jitter',
      'jitter_rap': 'jitter RAP',
      'jitter_ppq5': 'jitter PPQ5',
      'shimmer_local': 'shimmer',
      'shimmer_apq3': 'shimmer APQ3',
      'hnr': 'rapport harmoniques-bruit',
      'rms_energy': 'énergie vocale',
      'log_energy': 'énergie logarithmique',
      'spec_centroid_mean': 'centre spectral',
      'spec_bandwidth_mean': 'largeur de bande spectrale',
    },
  };

  static const _featureAliases = <String, Set<String>>{
    'jitter_local': {'jitter', 'jitter local', 'pitch regularity'},
    'jitter_rap': {'rap'},
    'jitter_ppq5': {'ppq5'},
    'shimmer_local': {'shimmer', 'volume stability'},
    'shimmer_apq3': {'apq3'},
    'shimmer_apq5': {'apq5'},
    'hnr': {'hnr', 'harmonics-to-noise', 'noise ratio'},
    'f0_mean': {'f0', 'pitch', 'average pitch', 'fundamental frequency'},
    'f0_std': {'pitch variation', 'pitch variability', 'f0 std'},
    'voiced_fraction': {'voiced fraction', 'voicing', 'voiced frame'},
    'rms_energy': {'energy', 'loudness', 'volume', 'rms'},
    'log_energy': {'log energy'},
    'peak_amplitude': {'peak amplitude', 'peak'},
    'zcr': {'zcr', 'zero crossing rate'},
    'spec_centroid_mean': {'spectral centroid', 'brightness'},
    'spec_bandwidth_mean': {
      'spectral bandwidth',
      'bandwidth',
      'frequency spread',
    },
  };

  static _GemmaSessionState _buildSessionState({
    required Map<String, double> features,
    required double riskScore,
    required String riskLevel,
    required String language,
    required Map<String, double> modelScores,
  }) {
    final allFeatures = Map<String, double>.from(features);

    final primaryFindings = <String>[];
    for (final key in [
      'jitter_local',
      'jitter_rap',
      'jitter_ppq5',
      'shimmer_local',
      'shimmer_apq3',
    ]) {
      final value = allFeatures[key];
      if (value == null) continue;
      final annotation = _annotateFeature(key, value);
      if (annotation.contains('ABOVE REF')) {
        primaryFindings.add(
          '${_featureLabelFor(key)}: ${VoiceMetricPolicy.formatValue(key, value)} $annotation',
        );
      }
    }

    if (primaryFindings.isEmpty) {
      for (final key in [
        'f0_std',
        'voiced_fraction',
        'rms_energy',
        'spec_centroid_mean',
      ]) {
        final value = allFeatures[key];
        if (value == null) continue;
        primaryFindings.add(
          '${_featureLabelFor(key)}: ${VoiceMetricPolicy.formatValue(key, value)}',
        );
      }
    }

    return _GemmaSessionState(
      language: language,
      riskLevel: riskLevel,
      riskScore: riskScore,
      allFeatures: allFeatures,
      primaryFindings: primaryFindings,
      modelScores: modelScores,
    );
  }

  static String _topSuggestedFeature(_GemmaSessionState state) {
    for (final key in [
      'jitter_local',
      'jitter_rap',
      'jitter_ppq5',
      'shimmer_local',
      'shimmer_apq3',
      'f0_std',
      'rms_energy',
      'spec_centroid_mean',
    ]) {
      final value = state.allFeatures[key];
      if (value == null) continue;
      if (VoiceMetricPolicy.isReferenceHigh(key, value)) return key;
    }
    return state.allFeatures.keys.firstOrNull ?? 'jitter_local';
  }

  static String _featureLabelFor(String feature) {
    if (_featureLabels.containsKey(feature)) {
      return _featureLabels[feature]!;
    }
    if (feature.startsWith('mfcc_')) return feature.replaceAll('_', ' ');
    if (feature.startsWith('spec_')) return feature.replaceAll('_', ' ');
    if (feature.startsWith('f0_')) return feature.replaceAll('_', ' ');
    return feature.replaceAll('_', ' ');
  }

  static String _localizedFeatureLabel(String feature, String language) {
    return _localizedFeatureLabels[language]?[feature] ??
        _localizedFeatureLabels['English']?[feature] ??
        (_featureLabels[feature] ?? feature);
  }

  static String _localizedResultQuestion(String riskLevel, String language) {
    return switch (language) {
      'Italiano' => switch (riskLevel) {
        'low' => 'Cosa nel mio test è risultato rassicurante?',
        'moderate' => 'Che cosa significa un risultato moderato?',
        _ => 'Che cosa significa un risultato elevato?',
      },
      '中文' => switch (riskLevel) {
        'low' => '这次结果里，哪些地方比较让人安心？',
        'moderate' => '中等风险结果代表什么？',
        _ => '较高风险结果代表什么？',
      },
      'Español' => switch (riskLevel) {
        'low' => '¿Qué parte de mi resultado fue tranquilizadora?',
        'moderate' => '¿Qué significa un resultado moderado?',
        _ => '¿Qué significa un resultado de riesgo elevado?',
      },
      'Français' => switch (riskLevel) {
        'low' => 'Qu’est-ce qui est rassurant dans mon résultat ?',
        'moderate' => 'Que signifie un résultat modéré ?',
        _ => 'Que signifie un résultat à risque élevé ?',
      },
      _ => switch (riskLevel) {
        'low' => 'What in my result looked reassuring?',
        'moderate' => 'What does my moderate result mean?',
        _ => 'What does this elevated result mean?',
      },
    };
  }

  static String _localizedFeatureQuestion(
    String featureLabel,
    String language,
  ) {
    return switch (language) {
      'Italiano' => 'Il mio valore di $featureLabel cosa significa per me?',
      '中文' => '我的$featureLabel数值对我意味着什么？',
      'Español' => '¿Qué significa para mí mi valor de $featureLabel?',
      'Français' => 'Que signifie pour moi ma valeur de $featureLabel ?',
      _ => 'What does my $featureLabel value mean for me?',
    };
  }

  static String _localizedLimitsQuestion(String language) {
    return switch (language) {
      'Italiano' => 'Che cosa non può dirmi con certezza questa app?',
      '中文' => '这个应用不能明确告诉我什么？',
      'Español' => '¿Qué cosas esta app no puede decirme con certeza?',
      'Français' =>
        'Qu’est-ce que cette app ne peut pas me dire avec certitude ?',
      _ => 'What can this app not tell me with certainty?',
    };
  }

  static String _localizedVoiceWhyQuestion(String language) {
    return switch (language) {
      'Italiano' => 'Perché la voce viene usata in questo tipo di screening?',
      '中文' => '为什么声音可以用于这类筛查？',
      'Español' => '¿Por qué la voz se usa en este tipo de cribado?',
      'Français' =>
        'Pourquoi la voix est-elle utilisée dans ce type de dépistage ?',
      _ => 'Why is voice used in this kind of screening?',
    };
  }

  static String _localizedRepeatLaterQuestion(String language) {
    return switch (language) {
      'Italiano' => 'Quando avrebbe senso ripetere il test in futuro?',
      '中文' => '以后在什么情况下适合再次检测？',
      'Español' => '¿Cuándo tendría sentido repetir la prueba más adelante?',
      'Français' =>
        'Dans quel cas serait-il utile de refaire ce test plus tard ?',
      _ => 'When would it make sense to check again later?',
    };
  }

  static String _localizedDoctorPrepQuestion(String language) {
    return switch (language) {
      'Italiano' => 'Come dovrei prepararmi a parlarne con un medico?',
      '中文' => '如果我要和医生讨论，这次结果该怎么表述？',
      'Español' => '¿Cómo debería prepararme para comentarlo con un médico?',
      'Français' =>
        'Comment devrais-je me préparer pour en parler à un médecin ?',
      _ => 'How should I prepare to discuss this with a doctor?',
    };
  }

  static String _localizedConfounderQuestion(String language) {
    return switch (language) {
      'Italiano' =>
        'Stanchezza, rumore o disidratazione possono aver influenzato il mio risultato?',
      '中文' => '疲劳、噪音或脱水会不会影响我的结果？',
      'Español' =>
        '¿El cansancio, el ruido o la deshidratación pudieron influir en mi resultado?',
      'Français' =>
        'La fatigue, le bruit ou la déshydratation ont-ils pu influencer mon résultat ?',
      _ => 'Could fatigue, noise, or dehydration have affected my result?',
    };
  }

  static _FollowUpIntent _classifyFollowUpIntent(String question) {
    final q = question.toLowerCase();
    final focused = _extractFeatureFocus(question);
    if (focused.isNotEmpty) return _FollowUpIntent.featureMeaning;
    if (_containsAny(q, const [
      'doctor',
      'medico',
      'médico',
      'médecin',
      '医生',
      '醫生',
      'neurologist',
      'neurologo',
      'neurólogo',
      'neurologue',
      'appointment',
      'visit',
      'consult',
    ])) {
      return _FollowUpIntent.nextSteps;
    }
    if (_containsAny(q, const [
      'fatigue',
      'dehydration',
      'noise',
      'microphone',
      'recording',
      'cold',
      'stress',
      'sleep',
      'tired',
      'environment',
      'stanchezza',
      'disidratazione',
      'rumore',
      'registrazione',
      'raffreddore',
      '疲劳',
      '脱水',
      '噪音',
      '录音',
      'cansancio',
      'deshidratación',
      'ruido',
      'grabación',
      'fatigue',
      'déshydratation',
      'bruit',
      'enregistrement',
    ])) {
      return _FollowUpIntent.confounders;
    }
    if (_containsAny(q, const [
      'accurate',
      'accuracy',
      'reliable',
      'trust',
      'limitation',
      'wrong',
      'false positive',
      'false negative',
      'dataset',
      'trained',
      'affidabile',
      'limite',
      'dataset',
      '训练',
      '准确',
      '局限',
      'confiable',
      'limitación',
      'fiable',
      'limite',
      'limitation',
    ])) {
      return _FollowUpIntent.limitations;
    }
    if (_containsAny(q, const [
      'why',
      'mean',
      'means',
      'result',
      'score',
      'elevated',
      'moderate',
      'low',
      'significa',
      'risultato',
      'punteggio',
      'elevato',
      'moderato',
      '结果',
      '分数',
      '代表',
      '较高风险',
      '中等风险',
      '是什么意思',
      'resultado',
      'puntuación',
      'moderado',
      'elevado',
      'résultat',
      'score',
      'modéré',
      'élevé',
      'signifie',
    ])) {
      return _FollowUpIntent.resultMeaning;
    }
    return _FollowUpIntent.general;
  }

  static bool _containsAny(String question, List<String> needles) {
    for (final needle in needles) {
      if (question.contains(needle)) return true;
    }
    return false;
  }

  static Set<String> _extractFeatureFocus(String question) {
    final q = question.toLowerCase();
    final focused = <String>{};
    final mfccMatch = RegExp(
      r'mfcc[_ ]?(\d{1,2})[_ ]?(mean|std)?',
    ).allMatches(q);
    for (final match in mfccMatch) {
      final index = match.group(1);
      final stat = match.group(2);
      if (index != null) {
        if (stat != null) {
          focused.add('mfcc_${index}_$stat');
        } else {
          focused.add('mfcc_${index}_mean');
          focused.add('mfcc_${index}_std');
        }
      }
    }
    for (final entry in _featureAliases.entries) {
      if (q.contains(entry.key)) {
        focused.add(entry.key);
        continue;
      }
      if (entry.value.any(q.contains)) focused.add(entry.key);
    }
    return focused;
  }

  static List<String> _selectFeatureLines({
    required _GemmaSessionState state,
    required _FollowUpIntent intent,
    required Set<String> focusedFeatures,
  }) {
    final keys = <String>[];
    if (focusedFeatures.isNotEmpty) {
      keys.addAll(focusedFeatures);
    } else if (intent == _FollowUpIntent.resultMeaning ||
        intent == _FollowUpIntent.general) {
      for (final key in [
        'jitter_local',
        'jitter_rap',
        'jitter_ppq5',
        'shimmer_local',
        'shimmer_apq3',
      ]) {
        final value = state.allFeatures[key];
        if (value == null) continue;
        if (_annotateFeature(key, value).contains('ABOVE REF')) keys.add(key);
      }
      if (keys.isEmpty) {
        keys.addAll(['f0_std', 'voiced_fraction', 'rms_energy']);
      }
    } else if (intent == _FollowUpIntent.confounders) {
      keys.addAll(['jitter_local', 'shimmer_local', 'rms_energy']);
    }

    final unique = <String>[];
    for (final key in keys) {
      if (!unique.contains(key) && state.allFeatures.containsKey(key)) {
        unique.add(key);
      }
    }

    return unique.take(5).map((key) {
      final value = state.allFeatures[key]!;
      final annotation = _annotateFeature(key, value);
      final label = _featureLabelFor(key);
      return '$label ($key): ${VoiceMetricPolicy.formatValue(key, value)}$annotation';
    }).toList();
  }

  static List<String> _buildPersonalizedInsights({
    required _GemmaSessionState state,
    required _FollowUpIntent intent,
    required Set<String> focusedFeatures,
  }) {
    if (focusedFeatures.isEmpty || intent != _FollowUpIntent.featureMeaning) {
      return const [];
    }

    final insights = <String>[];
    for (final feature in focusedFeatures.take(2)) {
      final value = state.allFeatures[feature];
      if (value == null) continue;

      final label = _featureLabelFor(feature);
      final annotation = _annotateFeature(feature, value);

      if (VoiceMetricPolicy.hasReferenceThreshold(feature)) {
        final status = annotation.isEmpty
            ? 'has no threshold annotation in the app'
            : annotation
                  .replaceAll('[', '')
                  .replaceAll(']', '')
                  .replaceAll(' vs ', ' compared with ');
        insights.add(
          'Your $label is ${VoiceMetricPolicy.formatValue(feature, value)} in this screening, and it is $status',
        );
      } else {
        insights.add(
          'Your $label is ${VoiceMetricPolicy.formatValue(feature, value)} in this screening, and this measure is interpreted in context rather than against a fixed healthy cutoff',
        );
      }

      insights.add(_relationToRiskState(feature, state));
      insights.addAll(_relatedFeatureComparisons(feature, state));
    }

    final unique = <String>[];
    for (final insight in insights) {
      if (!unique.contains(insight)) unique.add(insight);
    }
    return unique.take(4).toList();
  }

  static String _relationToRiskState(String feature, _GemmaSessionState state) {
    final value = state.allFeatures[feature];
    if (value == null) {
      return 'This measure still needs to be read alongside the rest of the voice profile.';
    }
    return 'This single measure does not determine the classifier result on its own; the screening result comes from the full model across many features.';
  }

  static List<String> _relatedFeatureComparisons(
    String feature,
    _GemmaSessionState state,
  ) {
    final related = switch (feature) {
      'jitter_local' ||
      'jitter_rap' ||
      'jitter_ppq5' => ['jitter_local', 'jitter_rap', 'jitter_ppq5'],
      'shimmer_local' || 'shimmer_apq3' => ['shimmer_local', 'shimmer_apq3'],
      'f0_mean' ||
      'f0_std' ||
      'voiced_fraction' => ['f0_mean', 'f0_std', 'voiced_fraction'],
      'rms_energy' || 'log_energy' => ['rms_energy', 'log_energy'],
      'spec_centroid_mean' ||
      'spec_bandwidth_mean' => ['spec_centroid_mean', 'spec_bandwidth_mean'],
      _ => const <String>[],
    };

    final lines = <String>[];
    for (final key in related) {
      if (key == feature) continue;
      final value = state.allFeatures[key];
      if (value == null) continue;
      final label = _featureLabelFor(key);
      final annotation = _annotateFeature(key, value);
      final suffix = annotation.isEmpty
          ? ''
          : ' (${annotation.replaceAll('[', '').replaceAll(']', '')})';
      lines.add(
        'Related measure: your $label is ${VoiceMetricPolicy.formatValue(key, value)}$suffix',
      );
    }
    return lines.take(2).toList();
  }

  static Future<List<KnowledgeCard>> _selectKnowledgeCards({
    required _GemmaSessionState state,
    required _FollowUpIntent intent,
    required Set<String> focusedFeatures,
  }) async {
    final allCards = await KnowledgeBaseService.loadScreeningCards();
    final cards = <KnowledgeCard>[];

    void addById(String id) {
      final card = allCards.where((c) => c.id == id).firstOrNull;
      if (card != null && !cards.any((c) => c.id == card.id)) cards.add(card);
    }

    addById('result_meaning.${state.riskLevel}');

    switch (intent) {
      case _FollowUpIntent.nextSteps:
        addById('doctor_visit.when_to_discuss');
        addById('limitations.attribution');
      case _FollowUpIntent.limitations:
        addById('limitations.dataset');
        addById('limitations.attribution');
      case _FollowUpIntent.confounders:
        addById('confounders.common');
      case _FollowUpIntent.resultMeaning:
        addById('limitations.full_model');
      case _FollowUpIntent.general:
        addById('screening_basics.scope');
      case _FollowUpIntent.featureMeaning:
        break;
    }

    for (final feature in focusedFeatures) {
      for (final card in allCards) {
        if (card.featureKeys.contains(feature) &&
            !cards.any((existing) => existing.id == card.id)) {
          cards.add(card);
        }
      }
    }

    if (focusedFeatures.isEmpty && intent == _FollowUpIntent.general) {
      addById('screening_basics.overview');
    }

    return cards.take(4).toList();
  }

  static String _annotateFeature(String feature, double value) {
    return VoiceMetricPolicy.annotation(feature, value);
  }

  static Map<String, String> _parseDoctorGuideUserContext(String? userNotes) {
    if (userNotes == null || userNotes.trim().isEmpty) return const {};
    final parsed = <String, String>{};
    for (final line in userNotes.split('\n')) {
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;
      final separator = trimmed.indexOf(':');
      if (separator <= 0 || separator == trimmed.length - 1) continue;
      final key = trimmed.substring(0, separator).trim();
      final value = trimmed.substring(separator + 1).trim();
      if (value.isNotEmpty) parsed[key] = value;
    }
    return parsed;
  }

  static Future<List<KnowledgeCard>> _selectDoctorGuideCards({
    required _GemmaSessionState state,
    required Map<String, String> userContext,
  }) async {
    final cards = await KnowledgeBaseService.loadDoctorCards();
    final relevantCards = <KnowledgeCard>[];

    void addById(String id) {
      final card = cards.where((c) => c.id == id).firstOrNull;
      if (card != null && !relevantCards.any((c) => c.id == card.id)) {
        relevantCards.add(card);
      }
    }

    addById('doctor_visit.opening_statement');
    addById('doctor_visit.${state.riskLevel}_guidance');
    addById('doctor_visit.why_bring_it_up');
    addById('doctor_visit.context_to_share');
    addById('doctor_visit.useful_questions');
    addById('doctor_visit.report_limits');
    addById('doctor_visit.shareable_tone');

    if (userContext.containsKey('noticed_changes')) {
      addById('doctor_visit.symptom_examples');
      addById('doctor_visit.family_observations');
    }
    if (userContext.containsKey('recording_factors')) {
      addById('doctor_visit.questions_about_context');
    }
    if (state.riskLevel != 'low') {
      addById('doctor_visit.repeat_screening_question');
      addById('doctor_visit.follow_up_goal');
    } else {
      addById('doctor_visit.if_result_conflicts_with_feelings');
    }

    return relevantCards;
  }

  static String _doctorGuideRiskInstructions(String riskLevel) {
    return switch (riskLevel) {
      'low' =>
        '- Low risk: keep the tone reassuring and baseline-oriented\n'
            '- Low risk: do not make doctor follow-up sound urgent or automatic\n'
            '- Low risk: if user context mentions symptoms or changes, frame the guide as a calm way to bring those concerns up',
      'moderate' =>
        '- Moderate risk: make the guide practical, watchful, and worth discussing\n'
            '- Moderate risk: include recording conditions and whether repeating the screening under better conditions might help\n'
            '- Moderate risk: keep the follow-up framing balanced rather than alarming',
      _ =>
        '- Elevated risk: make the guide direct, calm, and clinician-conversation ready\n'
            '- Elevated risk: keep the emphasis on timely discussion and practical next steps, not emergency language\n'
            '- Elevated risk: mention confounders without using them to dismiss the result',
    };
  }

  static List<String> _doctorGuideContextItems(
    Map<String, String> userContext,
  ) {
    final items = <String>[
      'This result came from an on-device voice screening app.',
      'The screening score came from 56 extracted voice markers.',
    ];

    final visitReason = userContext['visit_reason'];
    if (visitReason != null && visitReason.isNotEmpty) {
      items.add('Why I opened the app: $visitReason');
    }

    final noticedChanges = userContext['noticed_changes'];
    if (noticedChanges != null && noticedChanges.isNotEmpty) {
      items.add('Changes I noticed: $noticedChanges');
    }

    final recordingFactors = userContext['recording_factors'];
    if (recordingFactors != null && recordingFactors.isNotEmpty) {
      items.add('Recording conditions to mention: $recordingFactors');
    }

    return items;
  }

  static Future<DoctorDiscussionGuide> generateDoctorDiscussionGuide({
    required String language,
    String? userNotes,
    int? userAge,
  }) async {
    if (_engine == null || _lastSessionState == null) {
      throw StateError('No active session. Call analyze() first.');
    }

    final state = _lastSessionState!;
    final gemmaLanguage = LanguagePreferenceService.gemmaNameFor(language);
    final userContext = _parseDoctorGuideUserContext(userNotes);
    final relevantCards = await _selectDoctorGuideCards(
      state: state,
      userContext: userContext,
    );

    final referenceBlock = relevantCards
        .map((card) => '- ${card.id}: ${card.title} — ${card.body}')
        .join('\n');

    final systemPrompt =
        'LANGUAGE: You MUST respond in $gemmaLanguage. All text values in your JSON output must be in $gemmaLanguage.\n\n'
        'You are CogniTrace, an on-device screening assistant helping a user prepare for a doctor conversation.\n'
        'Rules:\n'
        '- This is a screening tool, never a diagnosis\n'
        '- Do not recommend medications, treatment plans, or prognosis\n'
        '- Do not claim that one biomarker caused the classifier result\n'
        '- Produce practical, calm, shareable language\n'
        '- Use the user context when present so the guide does not sound generic\n'
        '- Make the guide clearly different for low, moderate, and elevated risk results\n'
        '- Output valid JSON only with the required fields\n';

    final userPrompt = StringBuffer()
      ..writeln('CURRENT SCREENING RESULT:')
      ..writeln(
        '- risk: ${state.riskLevel} (${state.riskScore.toStringAsFixed(2)})',
      )
      ..writeln(
        '- user-facing basis: 56 acoustic voice markers extracted from a 60-second recording',
      )
      ..writeln('- this is not a diagnosis')
      ..writeln()
      ..writeln('RISK-SPECIFIC WRITING INSTRUCTIONS:')
      ..writeln(_doctorGuideRiskInstructions(state.riskLevel))
      ..writeln()
      ..writeln('REFERENCE CARDS:')
      ..writeln(referenceBlock)
      ..writeln();

    if (userAge != null) {
      userPrompt.writeln('USER CONTEXT:');
      userPrompt.writeln('- age: $userAge');
      userPrompt.writeln();
    }
    if (userNotes != null && userNotes.trim().isNotEmpty) {
      userPrompt.writeln('USER CONTEXT FIELDS:');
      userPrompt.writeln(
        '- visit_reason: why the user opened the app or wants help preparing',
      );
      userPrompt.writeln(
        '- noticed_changes: any voice, speech, swallowing, or movement changes the user noticed',
      );
      userPrompt.writeln(
        '- recording_factors: tired, sick/hoarse, noisy room, or similar recording conditions',
      );
      userPrompt.writeln();
      userPrompt.writeln('USER CONTEXT VALUES:');
      userPrompt.writeln(userNotes.trim());
      userPrompt.writeln();
    }

    userPrompt
      ..writeln('Return JSON with exactly these keys:')
      ..writeln('{')
      ..writeln('  "visit_reason": "...",')
      ..writeln('  "result_summary": "...",')
      ..writeln('  "questions_to_ask": ["..."],')
      ..writeln('  "context_to_share": ["..."],')
      ..writeln('  "caveats": ["..."]')
      ..writeln('}')
      ..writeln('Requirements:')
      ..writeln(
        '- Make the visit_reason sound natural for this specific risk level',
      )
      ..writeln(
        '- When visit_reason or noticed_changes are present, reflect them in the guide instead of generic filler',
      )
      ..writeln('- Keep the result summary practical and non-diagnostic')
      ..writeln(
        '- Do not mention model agreement, model names, classifier internals, ensemble models, or "3 models" in user-facing fields',
      )
      ..writeln(
        '- If explaining what the app measured, say it looked across 56 voice markers or a 56-marker voice profile',
      )
      ..writeln(
        '- Include 3 concise questions to ask, and make them specific to this risk level rather than generic',
      )
      ..writeln(
        '- Include context that may matter to a clinician, especially recording conditions, symptoms, and user notes',
      )
      ..writeln('- Include caveats about the app and the model limits')
      ..writeln(
        '- Keep the whole guide polished enough that a user could share it without rewriting it first',
      )
      ..writeln('- Do not mention treatment or medication')
      ..writeln('- Do not claim that one biomarker caused the result')
      ..writeln()
      ..writeln(
        'IMPORTANT: All JSON string values must be written in $gemmaLanguage.',
      );

    final chat = ChatSession(_engine!, systemPrompt: systemPrompt);
    final buffer = StringBuffer();
    await for (final chunk
        in chat
            .create(
              [LlamaTextContent(userPrompt.toString())],
              enableThinking: false,
              params: _defaultGenParams,
            )
            .timeout(_inferenceTimeout)) {
      final content = chunk.choices.firstOrNull?.delta.content;
      if (content != null) buffer.write(content);
    }

    final parsed = _parseJsonObject(buffer.toString());
    if (parsed != null) {
      final guide = DoctorDiscussionGuide.fromJson(parsed);
      if (guide.isComplete) return guide;
    }
    return _fallbackDoctorDiscussionGuide(
      riskLevel: state.riskLevel,
      riskScore: state.riskScore,
      language: language,
      userNotes: userNotes,
    );
  }

  @visibleForTesting
  static DoctorDiscussionGuide debugFallbackDoctorDiscussionGuide({
    required String riskLevel,
    required double riskScore,
    required String language,
    String? userNotes,
  }) {
    return _fallbackDoctorDiscussionGuide(
      riskLevel: riskLevel,
      riskScore: riskScore,
      language: language,
      userNotes: userNotes,
    );
  }

  static Map<String, dynamic>? _parseJsonObject(String response) {
    final start = response.indexOf('{');
    if (start < 0) return null;
    var depth = 0;
    for (var i = start; i < response.length; i++) {
      final char = response[i];
      if (char == '{') depth++;
      if (char == '}') {
        depth--;
        if (depth == 0) {
          final candidate = response.substring(start, i + 1);
          try {
            return jsonDecode(candidate) as Map<String, dynamic>;
          } catch (_) {
            return null;
          }
        }
      }
    }
    return null;
  }

  static DoctorDiscussionGuide _fallbackDoctorDiscussionGuide({
    required String riskLevel,
    required double riskScore,
    required String language,
    String? userNotes,
  }) {
    final userContext = _parseDoctorGuideUserContext(userNotes);

    final visitReason = switch (riskLevel) {
      'low' => switch (language) {
        'Italiano' =>
          userContext['visit_reason'] ??
              'Ho utilizzato uno strumento di screening vocale e vorrei capire come interpretare questo risultato.',
        '中文' => userContext['visit_reason'] ?? '我使用了一个语音筛查工具，想了解该如何理解这次结果。',
        'Español' =>
          userContext['visit_reason'] ??
              'Utilicé una herramienta de cribado por voz y me gustaría entender cómo interpretar este resultado.',
        'Français' =>
          userContext['visit_reason'] ??
              'J’ai utilisé un outil de dépistage vocal et je voudrais comprendre comment interpréter ce résultat.',
        _ =>
          userContext['visit_reason'] ??
              'I used a voice screening app and would like help understanding this result.',
      },
      'moderate' => switch (language) {
        'Italiano' =>
          userContext['visit_reason'] ??
              'Ho ricevuto un risultato di screening moderato e vorrei discutere se è opportuno un approfondimento.',
        '中文' => userContext['visit_reason'] ?? '我得到了一次中等风险筛查结果，想讨论是否需要进一步评估。',
        'Español' =>
          userContext['visit_reason'] ??
              'Recibí un resultado de cribado moderado y me gustaría comentar si conviene seguir evaluándolo.',
        'Français' =>
          userContext['visit_reason'] ??
              'J’ai reçu un résultat de dépistage modéré et je voudrais discuter de l’intérêt d’une évaluation complémentaire.',
        _ =>
          userContext['visit_reason'] ??
              'I received a moderate screening result and would like to discuss whether follow-up makes sense.',
      },
      _ => switch (language) {
        'Italiano' =>
          userContext['visit_reason'] ??
              'Ho ricevuto un risultato di screening elevato e vorrei capire quali passi successivi siano appropriati.',
        '中文' => userContext['visit_reason'] ?? '我得到了一次较高风险筛查结果，想了解下一步应该如何处理。',
        'Español' =>
          userContext['visit_reason'] ??
              'Recibí un resultado de cribado elevado y me gustaría entender qué seguimiento sería adecuado.',
        'Français' =>
          userContext['visit_reason'] ??
              'J’ai reçu un résultat de dépistage élevé et je voudrais comprendre quel suivi serait approprié.',
        _ =>
          userContext['visit_reason'] ??
              'I received an elevated screening result and would like to understand what follow-up would be appropriate.',
      },
    };

    final resultSummary = switch (riskLevel) {
      'low' =>
        'The app looked across 56 voice markers from this recording and found a low-risk pattern. It is best read as reassuring for this recording and can serve as a baseline if you decide to check again later.',
      'moderate' =>
        'The app looked across 56 voice markers from this recording and found a moderate-risk pattern. It may be worth discussing in context, especially if symptoms or changes are present.',
      _ =>
        'The app looked across 56 voice markers from this recording and found an elevated-risk pattern. It is not a diagnosis, but it is worth discussing with a clinician.',
    };

    final context = _doctorGuideContextItems(userContext);

    final questionsToAsk = switch (riskLevel) {
      'low' => const [
        'If I still notice changes in my voice or speech, what would be worth monitoring?',
        'Could tiredness, illness, or recording conditions have influenced this screening result?',
        'Would repeating the screening later be useful if my voice changes over time?',
      ],
      'moderate' => const [
        'Does this moderate screening result suggest any follow-up would be reasonable?',
        'Would you want me to repeat screening or have a speech or neurological assessment?',
        'Could fatigue, illness, or recording conditions have influenced this result?',
      ],
      _ => const [
        'Does this elevated screening result suggest that further evaluation would be appropriate?',
        'What kinds of next steps would usually make sense after a result like this?',
        'Could any non-neurological factors or recording conditions have influenced this result?',
      ],
    };

    return DoctorDiscussionGuide(
      visitReason: visitReason,
      resultSummary: resultSummary,
      questionsToAsk: questionsToAsk,
      contextToShare: context,
      caveats: const [
        'This app is a screening tool, not a diagnostic test.',
        'The model was trained on one research dataset.',
        'No single biomarker explains the result by itself.',
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Rehearsal (conversation practice) — public API
  // ---------------------------------------------------------------------------

  /// Start a new rehearsal session and stream the opening question.
  static Stream<String> startRehearsalStream({
    required DoctorDiscussionGuide guide,
    required String language,
  }) async* {
    await loadModel(); // idempotent + coalesces concurrent callers
    final gemmaLanguage = LanguagePreferenceService.gemmaNameFor(language);
    _rehearsalState = _RehearsalState(guide: guide, language: gemmaLanguage);
    final state = _rehearsalState!;
    final prompt = _buildPracticePrompt(state: state, isOpening: true);
    final chat = ChatSession(
      _engine!,
      systemPrompt: _practiceSystemPrompt(gemmaLanguage),
    );
    final buffer = StringBuffer();
    await for (final chunk
        in chat
            .create(
              [LlamaTextContent(prompt)],
              enableThinking: false,
              params: _defaultGenParams,
            )
            .timeout(_inferenceTimeout)) {
      final content = chunk.choices.firstOrNull?.delta.content;
      if (content != null) {
        buffer.write(content);
        yield content;
      }
    }
    state.history.add((
      role: 'model',
      text: _truncatePractice(buffer.toString().trim(), 200),
    ));
  }

  /// Continue the rehearsal with a user message and stream the next response.
  static Stream<String> continueRehearsalStream(String userMessage) async* {
    final state = _rehearsalState;
    if (_engine == null || state == null) {
      throw StateError(
        'No active rehearsal. Call startRehearsalStream() first.',
      );
    }
    state.topicsCovered.addAll(_detectPracticeTopics(userMessage, state.guide));
    final prompt = _buildPracticePrompt(state: state, userMessage: userMessage);
    state.history.add((
      role: 'user',
      text: _truncatePractice(userMessage, 200),
    ));
    final chat = ChatSession(
      _engine!,
      systemPrompt: _practiceSystemPrompt(state.language),
    );
    final buffer = StringBuffer();
    await for (final chunk
        in chat
            .create(
              [LlamaTextContent(prompt)],
              enableThinking: false,
              params: _defaultGenParams,
            )
            .timeout(_inferenceTimeout)) {
      final content = chunk.choices.firstOrNull?.delta.content;
      if (content != null) {
        buffer.write(content);
        yield content;
      }
    }
    state.history.add((
      role: 'model',
      text: _truncatePractice(buffer.toString().trim(), 200),
    ));
  }

  /// End the rehearsal and stream a summary. Clears rehearsal state on completion.
  static Stream<String> endRehearsalStream() async* {
    final state = _rehearsalState;
    if (_engine == null || state == null) {
      throw StateError(
        'No active rehearsal. Call startRehearsalStream() first.',
      );
    }
    final prompt = _buildPracticePrompt(state: state, isSummary: true);
    final chat = ChatSession(
      _engine!,
      systemPrompt: _practiceSystemPrompt(state.language),
    );
    await for (final chunk
        in chat
            .create(
              [LlamaTextContent(prompt)],
              enableThinking: false,
              params: _defaultGenParams,
            )
            .timeout(_inferenceTimeout)) {
      final content = chunk.choices.firstOrNull?.delta.content;
      if (content != null) yield content;
    }
    _rehearsalState = null;
  }

  /// Returns up to 4 suggested responses based on uncovered rehearsal topics.
  ///
  /// After [_softTurnLimit] exchanges, prepends a wrap-up hint. Always appends
  /// the end-practice chip as the last item.
  static List<String> suggestedRehearsalResponses(String language) {
    final state = _rehearsalState;
    if (state == null) return const [];
    final exchangeCount = state.history.where((m) => m.role == 'user').length;
    final suggestions = <String>[];
    if (exchangeCount >= _softTurnLimit) {
      suggestions.add(AppStrings.get('practice_end_prompt', language));
    }
    if (!state.topicsCovered.contains(RehearsalTopic.visitReason) &&
        state.guide.visitReason.isNotEmpty) {
      suggestions.add(_truncatePractice(state.guide.visitReason, 80));
    }
    if (!state.topicsCovered.contains(RehearsalTopic.screeningResult) &&
        state.guide.resultSummary.isNotEmpty) {
      suggestions.add(_truncatePractice(state.guide.resultSummary, 80));
    }
    if (!state.topicsCovered.contains(RehearsalTopic.nextSteps) &&
        state.guide.questionsToAsk.isNotEmpty) {
      suggestions.add(state.guide.questionsToAsk.first);
    }
    if (exchangeCount < _softTurnLimit) {
      suggestions.add(AppStrings.get('practice_end_prompt', language));
    }
    return suggestions.take(4).toList();
  }

  /// Reset any active rehearsal session without generating a summary.
  static void resetRehearsal() {
    _rehearsalState = null;
  }

  /// Whether a rehearsal session is currently active.
  static bool get hasActiveRehearsal => _rehearsalState != null;

  // ---------------------------------------------------------------------------
  // Rehearsal — private helpers
  // ---------------------------------------------------------------------------

  static String _practiceSystemPrompt(String language) {
    return 'LANGUAGE: You MUST respond in $language. Every word of your reply must be in $language.\n\n'
        'You help the user practice explaining their voice screening result before '
        'a doctor visit. You simulate the kinds of questions a clinician might ask, '
        'so the user can rehearse their answers.\n\n'
        'You are NOT a doctor. You cannot provide diagnoses, treatment plans, or '
        'medical advice. If the user asks for medical advice, remind them this is '
        'practice and redirect to their actual doctor.\n\n'
        'Your behavior:\n'
        '- Ask one question at a time, like a clinician would\n'
        '- Keep responses to 2-3 sentences\n'
        '- Help the user articulate their situation clearly\n'
        '- Guide the conversation toward topics from their preparation guide\n'
        '- Be warm and encouraging about their preparation\n'
        '- Always respond in $language, regardless of what language the instructions are in';
  }

  static String _buildPracticePrompt({
    required _RehearsalState state,
    String? userMessage,
    bool isOpening = false,
    bool isSummary = false,
  }) {
    final sessionState = _lastSessionState;
    final buffer = StringBuffer();

    if (isSummary) {
      buffer.writeln('CONVERSATION SUMMARY REQUEST:');
      buffer.writeln();
      final coveredNames = state.topicsCovered.map((t) => t.name).join(', ');
      final remainingTopics = RehearsalTopic.values
          .where((t) => !state.topicsCovered.contains(t))
          .map((t) => t.name)
          .toList();
      buffer.writeln(
        'Topics covered: ${coveredNames.isEmpty ? 'none' : coveredNames}',
      );
      buffer.writeln(
        'Topics not yet covered: '
        '${remainingTopics.isEmpty ? 'none' : remainingTopics.join(', ')}',
      );
      buffer.writeln();
      buffer.writeln(
        'Please provide a 3-5 bullet summary of the practice conversation, '
        'noting which topics were covered and which remain. '
        'Write the summary in ${state.language}.',
      );
      return buffer.toString();
    }

    if (sessionState != null) {
      buffer.writeln('SCREENING CONTEXT:');
      buffer.writeln(
        '- Risk level: ${sessionState.riskLevel} '
        '(score: ${sessionState.riskScore.toStringAsFixed(2)})',
      );
      buffer.writeln();
    }

    if (isOpening) {
      buffer.writeln('PRACTICE GUIDE:');
      buffer.writeln('- Visit reason: ${state.guide.visitReason}');
      if (state.guide.questionsToAsk.isNotEmpty) {
        buffer.writeln(
          '- First key question: ${state.guide.questionsToAsk.first}',
        );
      }
      buffer.writeln();
      buffer.writeln(
        'Begin the practice session. Respond in ${state.language}.',
      );
    } else {
      final allHistory = state.history;
      final maxMessages = _maxRehearsalHistoryTurns * 2;
      final truncated = allHistory.length > maxMessages;
      final windowHistory = truncated
          ? allHistory.sublist(allHistory.length - maxMessages)
          : allHistory;

      if (truncated) {
        buffer.writeln(
          'EARLIER IN CONVERSATION: Some earlier exchanges were omitted for brevity.',
        );
        buffer.writeln();
      }

      buffer.writeln('CONVERSATION SO FAR:');
      for (final turn in windowHistory) {
        buffer.writeln('[${turn.role}]: ${turn.text}');
      }
      buffer.writeln();

      final coveredNames = state.topicsCovered.map((t) => t.name).join(', ');
      final remainingNames = RehearsalTopic.values
          .where((t) => !state.topicsCovered.contains(t))
          .map((t) => t.name)
          .join(', ');
      if (coveredNames.isNotEmpty) {
        buffer.writeln('Topics discussed: $coveredNames');
      }
      if (remainingNames.isNotEmpty) {
        buffer.writeln('Topics remaining: $remainingNames');
      }
      buffer.writeln();

      buffer.writeln('USER: ${_truncatePractice(userMessage!, 200)}');
      buffer.writeln();
      buffer.writeln('Remember: respond in ${state.language} only.');
    }

    return buffer.toString();
  }

  static Set<RehearsalTopic> _detectPracticeTopics(
    String message,
    DoctorDiscussionGuide guide,
  ) {
    final lower = message.toLowerCase();
    final topics = <RehearsalTopic>{};
    if (lower.contains('screening') ||
        lower.contains('app') ||
        lower.contains('voice') ||
        lower.contains('reason')) {
      topics.add(RehearsalTopic.visitReason);
    }
    if (lower.contains('result') ||
        lower.contains('score') ||
        lower.contains('risk') ||
        lower.contains('elevated') ||
        lower.contains('moderate')) {
      topics.add(RehearsalTopic.screeningResult);
    }
    if (lower.contains('next') ||
        lower.contains('follow-up') ||
        lower.contains('follow up') ||
        lower.contains('repeat') ||
        lower.contains('refer')) {
      topics.add(RehearsalTopic.nextSteps);
    }
    return topics;
  }

  static String _truncatePractice(String text, int maxLength) {
    if (text.length <= maxLength) return text;
    return '${text.substring(0, maxLength)}...';
  }

  // ---------------------------------------------------------------------------
  // Rehearsal — debug/test helpers
  // ---------------------------------------------------------------------------

  @visibleForTesting
  static String debugPracticeSystemPrompt(String language) =>
      _practiceSystemPrompt(language);

  @visibleForTesting
  static String debugBuildPracticePrompt({
    required DoctorDiscussionGuide guide,
    required String language,
    required Map<String, double> features,
    required double riskScore,
    List<({String role, String text})> history = const [],
    String? userMessage,
    bool isOpening = false,
    bool isSummary = false,
    Set<RehearsalTopic> topicsCovered = const {},
    Map<String, double> modelScores = const {},
  }) {
    final riskLevel = riskScore >= 0.7
        ? 'elevated'
        : riskScore >= 0.4
        ? 'moderate'
        : 'low';
    _lastSessionState = _buildSessionState(
      features: features,
      riskScore: riskScore,
      riskLevel: riskLevel,
      language: language,
      modelScores: modelScores,
    );
    final tempState = _RehearsalState(
      guide: guide,
      language: LanguagePreferenceService.gemmaNameFor(language),
    );
    tempState.history.addAll(history);
    tempState.topicsCovered.addAll(topicsCovered);
    return _buildPracticePrompt(
      state: tempState,
      userMessage: userMessage,
      isOpening: isOpening,
      isSummary: isSummary,
    );
  }

  @visibleForTesting
  static Set<RehearsalTopic> debugDetectPracticeTopics(
    String message,
    DoctorDiscussionGuide guide,
  ) => _detectPracticeTopics(message, guide);

  @visibleForTesting
  static List<String> debugSuggestedPracticeResponses({
    required DoctorDiscussionGuide guide,
    required String language,
    Set<RehearsalTopic> topicsCovered = const {},
  }) {
    final tempState = _RehearsalState(
      guide: guide,
      language: LanguagePreferenceService.gemmaNameFor(language),
    );
    tempState.topicsCovered.addAll(topicsCovered);
    _rehearsalState = tempState;
    final result = suggestedRehearsalResponses(language);
    _rehearsalState = null;
    return result;
  }
}
