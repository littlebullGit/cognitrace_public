import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';

abstract final class AudioArchiveService {
  static const _audioDirectory = 'saved_audio_runs';

  /// Save audio and return a RELATIVE path (e.g. "saved_audio_runs/123.wav").
  /// Relative paths survive iOS container UUID changes.
  static Future<String> saveRunAudio({
    required String id,
    required ResultsAudioPayload payload,
  }) async {
    final directory = await _audioDirectoryPath();
    await directory.create(recursive: true);
    final file = File('${directory.path}${Platform.pathSeparator}$id.wav');
    final bytes = payload.wavBytes;
    if (bytes != null) {
      await file.writeAsBytes(bytes, flush: true);
      return '$_audioDirectory/$id.wav'; // Relative path
    }

    final pcm = payload.pcm;
    final sampleRate = payload.sampleRate;
    if (pcm == null || sampleRate == null) {
      throw StateError('Missing audio payload for local archiving.');
    }

    final wavBytes = _encodePcm16Wav(pcm, sampleRate);
    await file.writeAsBytes(wavBytes, flush: true);
    return '$_audioDirectory/$id.wav'; // Relative path
  }

  /// Resolve a stored path (relative or legacy absolute) to an absolute path.
  static Future<String> resolvePath(String storedPath) async {
    if (storedPath.isEmpty) return '';
    // Already absolute — legacy record
    if (storedPath.startsWith('/')) return storedPath;
    // Relative — resolve against current Documents directory
    final documents = await getApplicationDocumentsDirectory();
    return '${documents.path}${Platform.pathSeparator}$storedPath';
  }

  /// Try to fix a legacy absolute path by extracting the relative portion
  /// and checking if the file exists at the current Documents directory.
  static Future<String?> tryMigratePath(String absolutePath) async {
    if (absolutePath.isEmpty) return null;
    // Extract the relative part after "Documents/"
    final docsIndex = absolutePath.indexOf('Documents/');
    if (docsIndex < 0) return null;
    final relativePart = absolutePath.substring(
      docsIndex + 'Documents/'.length,
    );
    final resolved = await resolvePath(relativePart);
    if (await File(resolved).exists()) {
      return relativePart; // Return relative path — file exists at new location
    }
    return null;
  }

  static Future<Uint8List> readWavBytes(String storedPath) async {
    final path = await resolvePath(storedPath);
    return File(path).readAsBytes();
  }

  static Future<bool> hasAudio(String storedPath) async {
    if (storedPath.isEmpty) return false;
    final path = await resolvePath(storedPath);
    return File(path).exists();
  }

  static Future<void> deleteAudio(String storedPath) async {
    if (storedPath.isEmpty) return;
    final path = await resolvePath(storedPath);
    final file = File(path);
    if (await file.exists()) {
      await file.delete();
    }
  }

  static Future<void> deleteAllAudio() async {
    final directory = await _audioDirectoryPath();
    if (await directory.exists()) {
      await directory.delete(recursive: true);
    }
  }

  static Future<Directory> _audioDirectoryPath() async {
    final documents = await getApplicationDocumentsDirectory();
    return Directory(
      '${documents.path}${Platform.pathSeparator}$_audioDirectory',
    );
  }

  static Uint8List _encodePcm16Wav(Float32List pcm, double sampleRate) {
    final dataLength = pcm.length * 2;
    final byteData = ByteData(44 + dataLength);

    void writeAscii(int offset, String value) {
      for (var i = 0; i < value.length; i++) {
        byteData.setUint8(offset + i, value.codeUnitAt(i));
      }
    }

    writeAscii(0, 'RIFF');
    byteData.setUint32(4, 36 + dataLength, Endian.little);
    writeAscii(8, 'WAVE');
    writeAscii(12, 'fmt ');
    byteData.setUint32(16, 16, Endian.little);
    byteData.setUint16(20, 1, Endian.little);
    byteData.setUint16(22, 1, Endian.little);
    byteData.setUint32(24, sampleRate.round(), Endian.little);
    byteData.setUint32(28, sampleRate.round() * 2, Endian.little);
    byteData.setUint16(32, 2, Endian.little);
    byteData.setUint16(34, 16, Endian.little);
    writeAscii(36, 'data');
    byteData.setUint32(40, dataLength, Endian.little);

    var offset = 44;
    for (final sample in pcm) {
      final clamped = sample.clamp(-1.0, 1.0);
      final intValue = (clamped * 32767).round();
      byteData.setInt16(offset, intValue, Endian.little);
      offset += 2;
    }

    return byteData.buffer.asUint8List();
  }

  static Uint8List encodePcm16Wav(Float32List pcm, double sampleRate) {
    return _encodePcm16Wav(pcm, sampleRate);
  }

  /// Decode a 16-bit PCM WAV file into a Float32List of samples.
  static Float32List decodePcmFromWav(Uint8List bytes) {
    final data = ByteData.sublistView(bytes);
    // Skip to 'data' chunk.
    var offset = 12;
    int? channelCount;
    int? dataOffset;
    int? dataLength;
    while (offset + 8 <= bytes.length) {
      final chunkId = String.fromCharCodes(bytes.sublist(offset, offset + 4));
      final chunkSize = data.getUint32(offset + 4, Endian.little);
      if (chunkId == 'fmt ') {
        channelCount = data.getUint16(offset + 8 + 2, Endian.little);
      } else if (chunkId == 'data') {
        dataOffset = offset + 8;
        dataLength = chunkSize;
        break;
      }
      offset += 8 + chunkSize + (chunkSize.isOdd ? 1 : 0);
    }
    channelCount ??= 1;
    if (dataOffset == null || dataLength == null) {
      throw const FormatException('No data chunk in WAV');
    }
    final frameCount = dataLength ~/ (channelCount * 2);
    final pcm = Float32List(frameCount);
    var cursor = dataOffset;
    for (var i = 0; i < frameCount; i++) {
      double mixed = 0;
      for (var ch = 0; ch < channelCount; ch++) {
        mixed += data.getInt16(cursor, Endian.little) / 32768.0;
        cursor += 2;
      }
      pcm[i] = mixed / channelCount;
    }
    return pcm;
  }
}

class ResultsAudioPayload {
  const ResultsAudioPayload({this.pcm, this.sampleRate, this.wavBytes});

  final Float32List? pcm;
  final double? sampleRate;
  final Uint8List? wavBytes;
}
