import 'dart:typed_data';

import 'package:flutter/services.dart';

const _channel = MethodChannel('com.cognitrace/audio');

class FeatureExtractionResult {
  const FeatureExtractionResult({required this.features, required this.trace});

  final Map<String, double> features;
  final Map<String, double> trace;
}

Future<void> startRecording() async {
  try {
    await _channel.invokeMethod<void>('startRecording');
  } on PlatformException catch (e) {
    _rethrow(e);
  }
}

Future<Float32List> stopRecording() async {
  try {
    final raw = await _channel.invokeMethod<dynamic>('stopRecording');
    if (raw is Float32List) return raw;
    if (raw is Uint8List) return raw.buffer.asFloat32List();
    throw AudioHardwareException(
      'Unexpected stopRecording return type: ${raw.runtimeType}',
    );
  } on PlatformException catch (e) {
    _rethrow(e);
  }
}

Future<double> getAudioLevel() async {
  try {
    return await _channel.invokeMethod<double>('getAudioLevel') ?? 0.0;
  } on PlatformException catch (e) {
    _rethrow(e);
  }
}

Future<void> playAudioFile(String path) async {
  try {
    await _channel.invokeMethod<void>('playAudioFile', {'path': path});
  } on PlatformException catch (e) {
    _rethrow(e);
  }
}

Future<void> pauseAudioPlayback() async {
  try {
    await _channel.invokeMethod<void>('pauseAudioPlayback');
  } on PlatformException catch (e) {
    _rethrow(e);
  }
}

Future<void> stopAudioPlayback() async {
  try {
    await _channel.invokeMethod<void>('stopAudioPlayback');
  } on PlatformException catch (e) {
    _rethrow(e);
  }
}

Future<void> restartAudioPlayback(String path) async {
  try {
    await _channel.invokeMethod<void>('restartAudioPlayback', {'path': path});
  } on PlatformException catch (e) {
    _rethrow(e);
  }
}

Future<FeatureExtractionResult> extractFeatures({
  required Float32List pcm,
  double sampleRate = 16000,
}) async {
  try {
    final raw = await _channel.invokeMapMethod<Object?, Object?>(
      'extractFeatures',
      {'pcm': pcm, 'sampleRate': sampleRate},
    );

    if (raw == null) {
      throw const AudioHardwareException(
        'Feature extraction returned no data.',
      );
    }

    final featuresRaw = raw['features'];
    final traceRaw = raw['trace'];
    if (featuresRaw is! Map<Object?, Object?> ||
        traceRaw is! Map<Object?, Object?>) {
      throw const AudioHardwareException(
        'Feature extraction returned malformed payload.',
      );
    }

    Map<String, double> convert(Map<Object?, Object?> source) {
      return source.map((key, value) {
        final name = key as String;
        final number = value as num;
        return MapEntry(name, number.toDouble());
      });
    }

    return FeatureExtractionResult(
      features: convert(featuresRaw),
      trace: convert(traceRaw),
    );
  } on PlatformException catch (e) {
    _rethrow(e);
  }
}

Future<FeatureExtractionResult> extractFeaturesFromWavBytes({
  required Uint8List wavBytes,
}) async {
  try {
    final raw = await _channel.invokeMapMethod<Object?, Object?>(
      'extractFeaturesFromWavBytes',
      {'wavBytes': wavBytes},
    );

    if (raw == null) {
      throw const AudioHardwareException(
        'Feature extraction returned no data.',
      );
    }

    final featuresRaw = raw['features'];
    final traceRaw = raw['trace'];
    if (featuresRaw is! Map<Object?, Object?> ||
        traceRaw is! Map<Object?, Object?>) {
      throw const AudioHardwareException(
        'Feature extraction returned malformed payload.',
      );
    }

    Map<String, double> convert(Map<Object?, Object?> source) {
      return source.map((key, value) {
        final name = key as String;
        final number = value as num;
        return MapEntry(name, number.toDouble());
      });
    }

    return FeatureExtractionResult(
      features: convert(featuresRaw),
      trace: convert(traceRaw),
    );
  } on PlatformException catch (e) {
    _rethrow(e);
  }
}

Never _rethrow(PlatformException e) {
  if (e.code == 'PERMISSION_DENIED') {
    throw AudioPermissionException(e.message ?? 'Microphone permission denied');
  }
  throw AudioHardwareException(e.message ?? 'Audio hardware error (${e.code})');
}

class AudioPermissionException implements Exception {
  const AudioPermissionException(this.message);
  final String message;
  @override
  String toString() => 'AudioPermissionException: $message';
}

class AudioHardwareException implements Exception {
  const AudioHardwareException(this.message);
  final String message;
  @override
  String toString() => 'AudioHardwareException: $message';
}
