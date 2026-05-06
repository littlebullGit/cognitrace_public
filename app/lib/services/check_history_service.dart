import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

import '../models/check_record.dart';
import 'audio_archive_service.dart';

/// Persists check history in the app's Documents directory.
///
/// - Storage: `check_history.json` in Documents (fast reads/writes)
/// - Uninstall: iOS removes the app container and this file with it
/// - Audio file paths are validated — missing files get paths cleared
abstract final class CheckHistoryService {
  static const _fileName = 'check_history.json';

  static Future<List<CheckRecord>> load() async {
    final file = await _historyFile();
    String? raw;

    if (await file.exists()) {
      raw = await file.readAsString();
      debugPrint(
        'CheckHistory: Documents file exists, ${raw.length} bytes, path: ${file.path}',
      );
    } else {
      debugPrint('CheckHistory: Documents file NOT found at ${file.path}');
    }

    if (raw == null || raw.trim().isEmpty) return [];

    final records = CheckRecord.decodeList(raw);
    debugPrint('CheckHistory: decoded ${records.length} records');

    // Fix audio paths: migrate legacy absolute paths to relative,
    // and clear genuinely missing files.
    var didFix = false;
    final fixed = <CheckRecord>[];
    for (final record in records) {
      if (record.audioFilePath.isNotEmpty) {
        // Check if audio exists (handles both relative and absolute paths).
        final exists = await AudioArchiveService.hasAudio(record.audioFilePath);
        if (exists) {
          fixed.add(record);
        } else if (record.audioFilePath.startsWith('/')) {
          // Legacy absolute path — try migrating to relative.
          final migrated = await AudioArchiveService.tryMigratePath(
            record.audioFilePath,
          );
          if (migrated != null) {
            debugPrint(
              'CheckHistory: migrated path for ${record.id}: $migrated',
            );
            fixed.add(record.copyWith(audioFilePath: migrated));
            didFix = true;
          } else {
            debugPrint(
              'CheckHistory: audio gone for ${record.id}: ${record.audioFilePath}',
            );
            fixed.add(record.copyWith(audioFilePath: ''));
            didFix = true;
          }
        } else {
          debugPrint(
            'CheckHistory: audio missing for ${record.id}: ${record.audioFilePath}',
          );
          fixed.add(record.copyWith(audioFilePath: ''));
          didFix = true;
        }
      } else {
        fixed.add(record);
      }
    }

    if (didFix) await _writeAll(fixed);

    fixed.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return fixed;
  }

  static Future<void> save(CheckRecord record) async {
    final records = await load();
    final filtered = records
        .where((existing) => existing.id != record.id)
        .toList();
    filtered.insert(0, record);
    await _writeAll(filtered);
  }

  static Future<void> delete(String id) async {
    final records = await load();
    final filtered = records.where((record) => record.id != id).toList();
    await _writeAll(filtered);
  }

  static Future<void> deleteAll() async {
    final file = await _historyFile();
    if (await file.exists()) {
      await file.delete();
    }
  }

  static Future<void> _writeAll(List<CheckRecord> records) async {
    final encoded = CheckRecord.encodeList(records);

    final file = await _historyFile();
    await file.parent.create(recursive: true);
    await file.writeAsString(encoded, flush: true);
  }

  static Future<File> _historyFile() async {
    final directory = await getApplicationDocumentsDirectory();
    return File('${directory.path}${Platform.pathSeparator}$_fileName');
  }
}
