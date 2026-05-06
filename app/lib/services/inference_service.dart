import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

class InferenceResult {
  const InferenceResult({
    required this.riskScore,
    required this.riskLabel,
    required this.featureVector,
    required this.modelName,
    required this.modelScores,
  });

  final double riskScore;
  final String riskLabel;
  final List<double> featureVector;
  final String modelName;
  final Map<String, double> modelScores;
}

class InferenceService {
  static const _modelVersion = 'swift-trained-v3-56f-799';
  static const _scalerAsset = 'assets/models/scaler_params.json';
  static const _xgbAsset = 'assets/models/xgb_model.onnx';
  static const _lgbAsset = 'assets/models/lgb_model_mobile.onnx';
  static const _cbAsset = 'assets/models/cb_model_mobile.onnx';

  static final OnnxRuntime _onnx = OnnxRuntime();

  static OrtSession? _xgbSession;
  static OrtSession? _lgbSession;
  static OrtSession? _cbSession;
  static _ScalerParams? _scalerParams;

  static Future<void> reset() async {
    await _xgbSession?.close();
    await _lgbSession?.close();
    await _cbSession?.close();
    _xgbSession = null;
    _lgbSession = null;
    _cbSession = null;
    _scalerParams = null;
    rootBundle.evict(_scalerAsset);
  }

  static Future<InferenceResult> runRiskInference(
    Map<String, double> extractedFeatures,
  ) async {
    final scaler = await _loadScalerParams();
    final sessions = await _loadSessions();

    final normalized = scaler.normalize(extractedFeatures);
    final inputTensor = await OrtValue.fromList(normalized, [
      1,
      normalized.length,
    ]);

    try {
      final xgbScore = await _runTensorProbabilityModel(
        session: sessions.xgb,
        inputTensor: inputTensor,
      );
      final lgbScore = await _runTensorProbabilityModel(
        session: sessions.lgb,
        inputTensor: inputTensor,
      );
      final cbScore = await _runTensorProbabilityModel(
        session: sessions.cb,
        inputTensor: inputTensor,
      );
      final riskScore = (xgbScore + lgbScore + cbScore) / 3;
      final modelScores = {
        'xgb_probability': xgbScore,
        'lgb_probability': lgbScore,
        'cb_probability': cbScore,
      };

      return InferenceResult(
        riskScore: riskScore,
        riskLabel: labelFor(riskScore),
        featureVector: normalized,
        modelName: 'xgb+lgb+cb ensemble',
        modelScores: modelScores,
      );
    } finally {
      inputTensor.dispose();
    }
  }

  static Future<_ScalerParams> _loadScalerParams() async {
    if (_scalerParams != null) return _scalerParams!;
    rootBundle.evict(_scalerAsset);
    final raw = await rootBundle.loadString(_scalerAsset);
    final json = jsonDecode(raw) as Map<String, dynamic>;
    _scalerParams = _ScalerParams.fromJson(json);
    return _scalerParams!;
  }

  static Future<_EnsembleSessions> _loadSessions() async {
    _xgbSession ??= await _loadVersionedSession(_xgbAsset);
    _lgbSession ??= await _loadVersionedSession(_lgbAsset);
    _cbSession ??= await _loadVersionedSession(_cbAsset);
    return _EnsembleSessions(
      xgb: _xgbSession!,
      lgb: _lgbSession!,
      cb: _cbSession!,
    );
  }

  static Future<OrtSession> _loadVersionedSession(String assetPath) async {
    final tempDir = await getTemporaryDirectory();
    final fileName = assetPath.split('/').last;
    final versionedName = '$_modelVersion-$fileName';
    final localPath = '${tempDir.path}${Platform.pathSeparator}$versionedName';
    final file = File(localPath);

    final data = await rootBundle.load(assetPath);
    await file.parent.create(recursive: true);
    await file.writeAsBytes(data.buffer.asUint8List(), flush: true);

    return _onnx.createSession(localPath);
  }

  static Future<double> _runTensorProbabilityModel({
    required OrtSession session,
    required OrtValue inputTensor,
  }) async {
    final outputs = await session.run({'features': inputTensor});
    final probabilitiesValue = outputs['probabilities'];
    if (probabilitiesValue == null) {
      throw const InferenceException('Model did not return probabilities.');
    }

    final probabilities = await probabilitiesValue.asFlattenedList();
    return _extractPositiveClassProbability(probabilities);
  }

  static double _extractPositiveClassProbability(List<dynamic> probabilities) {
    if (probabilities.length < 2) {
      throw InferenceException(
        'Expected 2-class probability output, got ${probabilities.length}.',
      );
    }
    final positive = probabilities[1] as num;
    return positive.toDouble().clamp(0.0, 1.0);
  }

  static String labelFor(double riskScore) {
    if (riskScore >= 0.7) return 'elevated';
    if (riskScore >= 0.4) return 'moderate';
    return 'low';
  }
}

class _EnsembleSessions {
  const _EnsembleSessions({
    required this.xgb,
    required this.lgb,
    required this.cb,
  });

  final OrtSession xgb;
  final OrtSession lgb;
  final OrtSession cb;
}

class _ScalerParams {
  const _ScalerParams({
    required this.featureNames,
    required this.mean,
    required this.scale,
  });

  final List<String> featureNames;
  final List<double> mean;
  final List<double> scale;

  factory _ScalerParams.fromJson(Map<String, dynamic> json) {
    return _ScalerParams(
      featureNames: (json['feature_names'] as List<dynamic>).cast<String>(),
      mean: (json['mean'] as List<dynamic>)
          .map((v) => (v as num).toDouble())
          .toList(),
      scale: (json['scale'] as List<dynamic>)
          .map((v) => (v as num).toDouble())
          .toList(),
    );
  }

  List<double> normalize(Map<String, double> extractedFeatures) {
    return List<double>.generate(featureNames.length, (index) {
      final raw = extractedFeatures[featureNames[index]] ?? 0.0;
      final denom = scale[index];
      if (denom == 0) return 0.0;
      return (raw - mean[index]) / denom;
    });
  }
}

class InferenceException implements Exception {
  const InferenceException(this.message);

  final String message;

  @override
  String toString() => 'InferenceException: $message';
}
