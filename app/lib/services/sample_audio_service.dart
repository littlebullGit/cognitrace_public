import 'dart:typed_data';

import 'package:flutter/services.dart';

class SampleAudioClip {
  const SampleAudioClip({
    required this.title,
    required this.label,
    required this.taskAssetPaths,
  });

  final String title;
  final String label;

  /// One asset path per voice task: [vowel, rhythm, speech].
  final List<String> taskAssetPaths;
}

class SampleAudioPayload {
  const SampleAudioPayload({
    required this.taskPcmList,
    required this.combinedPcm,
    required this.sampleRate,
    required this.combinedWavBytes,
    required this.taskSampleLengths,
  });

  /// Per-task PCM buffers for per-task inference.
  final List<Float32List> taskPcmList;

  /// Concatenated PCM for WAV archiving.
  final Float32List combinedPcm;
  final double sampleRate;

  /// WAV-encoded concatenated audio for saving.
  final Uint8List combinedWavBytes;

  /// Sample count for each task (for rerun reconstruction).
  final List<int> taskSampleLengths;
}

abstract final class SampleAudioService {
  static const pd = SampleAudioClip(
    title: 'Reference positive sample',
    label: 'PD',
    taskAssetPaths: [
      'assets/reference_audio/pd_vowel.wav',
      'assets/reference_audio/pd_rhythm.wav',
      'assets/reference_audio/pd_speech.wav',
    ],
  );

  static const control = SampleAudioClip(
    title: 'Reference control sample',
    label: 'Control',
    taskAssetPaths: [
      'assets/reference_audio/control_vowel.wav',
      'assets/reference_audio/control_rhythm.wav',
      'assets/reference_audio/control_speech.wav',
    ],
  );

  static const clips = [pd, control];

  static Future<SampleAudioPayload> loadClip(SampleAudioClip clip) async {
    final taskPcmList = <Float32List>[];
    double? sampleRate;

    for (final path in clip.taskAssetPaths) {
      final data = await rootBundle.load(path);
      final bytes = data.buffer.asUint8List();
      final decoded = _decodeWav(bytes);
      taskPcmList.add(decoded.pcm);
      sampleRate ??= decoded.sampleRate;
    }

    // Concatenate for archiving.
    final combinedLength = taskPcmList.fold<int>(
      0,
      (sum, pcm) => sum + pcm.length,
    );
    final combined = Float32List(combinedLength);
    var offset = 0;
    for (final pcm in taskPcmList) {
      combined.setAll(offset, pcm);
      offset += pcm.length;
    }

    final wavBytes = _encodePcm16Wav(combined, sampleRate!);
    final taskSampleLengths = taskPcmList.map((p) => p.length).toList();

    return SampleAudioPayload(
      taskPcmList: taskPcmList,
      combinedPcm: combined,
      sampleRate: sampleRate,
      combinedWavBytes: wavBytes,
      taskSampleLengths: taskSampleLengths,
    );
  }

  static _DecodedWav _decodeWav(Uint8List bytes) {
    final data = ByteData.sublistView(bytes);
    if (bytes.length < 44) {
      throw const FormatException('WAV file is too short.');
    }
    if (_ascii(bytes, 0, 4) != 'RIFF' || _ascii(bytes, 8, 4) != 'WAVE') {
      throw const FormatException('Unsupported WAV container.');
    }

    int offset = 12;
    int? channelCount;
    int? sampleRate;
    int? bitsPerSample;
    int? dataOffset;
    int? dataLength;

    while (offset + 8 <= bytes.length) {
      final chunkId = _ascii(bytes, offset, 4);
      final chunkSize = data.getUint32(offset + 4, Endian.little);
      final chunkDataOffset = offset + 8;

      if (chunkId == 'fmt ' && chunkDataOffset + chunkSize <= bytes.length) {
        final audioFormat = data.getUint16(chunkDataOffset, Endian.little);
        channelCount = data.getUint16(chunkDataOffset + 2, Endian.little);
        sampleRate = data.getUint32(chunkDataOffset + 4, Endian.little);
        bitsPerSample = data.getUint16(chunkDataOffset + 14, Endian.little);
        if (audioFormat != 1) {
          throw const FormatException('Only PCM WAV is supported.');
        }
      } else if (chunkId == 'data') {
        dataOffset = chunkDataOffset;
        dataLength = chunkSize;
        break;
      }

      offset = chunkDataOffset + chunkSize + (chunkSize.isOdd ? 1 : 0);
    }

    if (channelCount == null ||
        sampleRate == null ||
        bitsPerSample == null ||
        dataOffset == null ||
        dataLength == null) {
      throw const FormatException('Incomplete WAV file.');
    }
    if (bitsPerSample != 16) {
      throw const FormatException('Only 16-bit PCM WAV is supported.');
    }

    final frameCount = dataLength ~/ (channelCount * 2);
    final pcm = Float32List(frameCount);
    var cursor = dataOffset;
    for (var frame = 0; frame < frameCount; frame++) {
      double mixed = 0;
      for (var channel = 0; channel < channelCount; channel++) {
        mixed += data.getInt16(cursor, Endian.little) / 32768.0;
        cursor += 2;
      }
      pcm[frame] = (mixed / channelCount).toDouble();
    }

    return _DecodedWav(pcm: pcm, sampleRate: sampleRate.toDouble());
  }

  /// Minimal PCM-to-WAV encoder (16-bit mono).
  static Uint8List _encodePcm16Wav(Float32List pcm, double sampleRate) {
    final sr = sampleRate.round();
    final dataLength = pcm.length * 2;
    final fileLength = 44 + dataLength;
    final bytes = ByteData(fileLength);

    void writeAscii(int offset, String s) {
      for (var i = 0; i < s.length; i++) {
        bytes.setUint8(offset + i, s.codeUnitAt(i));
      }
    }

    writeAscii(0, 'RIFF');
    bytes.setUint32(4, fileLength - 8, Endian.little);
    writeAscii(8, 'WAVE');
    writeAscii(12, 'fmt ');
    bytes.setUint32(16, 16, Endian.little);
    bytes.setUint16(20, 1, Endian.little); // PCM
    bytes.setUint16(22, 1, Endian.little); // mono
    bytes.setUint32(24, sr, Endian.little);
    bytes.setUint32(28, sr * 2, Endian.little); // byte rate
    bytes.setUint16(32, 2, Endian.little); // block align
    bytes.setUint16(34, 16, Endian.little); // bits per sample
    writeAscii(36, 'data');
    bytes.setUint32(40, dataLength, Endian.little);

    var cursor = 44;
    for (final sample in pcm) {
      final clamped = sample.clamp(-1.0, 1.0);
      bytes.setInt16(cursor, (clamped * 32767).round(), Endian.little);
      cursor += 2;
    }

    return bytes.buffer.asUint8List();
  }

  static String _ascii(Uint8List bytes, int start, int length) {
    return String.fromCharCodes(bytes.sublist(start, start + length));
  }
}

class _DecodedWav {
  const _DecodedWav({required this.pcm, required this.sampleRate});
  final Float32List pcm;
  final double sampleRate;
}
