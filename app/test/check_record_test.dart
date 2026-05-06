import 'package:flutter_test/flutter_test.dart';
import 'package:cognitrace/models/check_record.dart';

void main() {
  group('CheckRecord serialization', () {
    test('toJson includes all fields', () {
      final record = CheckRecord(
        id: 'test-1',
        timestamp: DateTime.utc(2026, 4, 11),
        source: 'analysis',
        audioFilePath: '/path/to/audio.wav',
        riskLabel: 'low',
        riskScore: 0.25,
        modelName: 'ensemble_v1',
        modelScores: {'xgb': 0.22, 'lgb': 0.28},
        featureSummary: {'f0_mean': 120.0},
        trace: {'inference_ms': 15.0},
        name: 'John',
        age: 65,
        notes: 'First check',
      );
      final json = record.toJson();
      expect(json['id'], 'test-1');
      expect(json['riskLabel'], 'low');
      expect(json['riskScore'], 0.25);
      expect(json['name'], 'John');
      expect(json['age'], 65);
      expect(json['notes'], 'First check');
      expect(json['audioFilePath'], '/path/to/audio.wav');
    });

    test('toJson omits null optional fields', () {
      final record = CheckRecord(
        id: 'test-2',
        timestamp: DateTime.utc(2026, 4, 11),
        source: 'analysis',
        audioFilePath: '/path/to/audio.wav',
        riskLabel: 'low',
        riskScore: 0.1,
        modelName: 'ensemble_v1',
        modelScores: {},
        featureSummary: {},
        trace: {},
      );
      final json = record.toJson();
      expect(json.containsKey('name'), isFalse);
      expect(json.containsKey('age'), isFalse);
      expect(json.containsKey('notes'), isFalse);
      // featureError is always included (even if null)
      expect(json['featureError'], isNull);
    });

    test('fromJson loads record with name/age/notes', () {
      final json = {
        'id': 'test-3',
        'timestamp': '2026-04-11T00:00:00.000Z',
        'source': 'analysis',
        'audioFilePath': '/path/to/audio.wav',
        'riskLabel': 'moderate',
        'riskScore': 0.55,
        'modelName': 'ensemble_v1',
        'modelScores': {'xgb': 0.52, 'lgb': 0.58},
        'featureSummary': {'f0_mean': 125.0},
        'trace': {'inference_ms': 18.0},
        'name': 'Maria',
        'age': 72,
        'notes': 'Repeat in 3 months',
      };
      final record = CheckRecord.fromJson(json);
      expect(record.name, 'Maria');
      expect(record.age, 72);
      expect(record.notes, 'Repeat in 3 months');
      expect(record.riskLabel, 'moderate');
    });

    test('fromJson handles old records without name/age/notes', () {
      final json = {
        'id': 'old-1',
        'timestamp': '2026-01-01T00:00:00.000Z',
        'source': 'bundled_sample',
        'audioFilePath': '/bundled/sample.wav',
        'riskLabel': 'low',
        'riskScore': 0.15,
        'modelName': 'ensemble_v1',
        'modelScores': <String, double>{},
        'featureSummary': <String, double>{},
        'trace': <String, double>{},
      };
      final record = CheckRecord.fromJson(json);
      expect(record.name, isNull);
      expect(record.age, isNull);
      expect(record.notes, isNull);
      expect(record.id, 'old-1');
    });

    test('roundtrip toJson/fromJson preserves all data', () {
      final original = CheckRecord(
        id: 'rt-1',
        timestamp: DateTime.utc(2026, 4, 11, 14, 30),
        source: 'analysis',
        audioFilePath: '/path/to/audio.wav',
        riskLabel: 'elevated',
        riskScore: 0.85,
        modelName: 'ensemble_v1',
        modelScores: {'xgb': 0.82, 'lgb': 0.88, 'cb': 0.85},
        featureSummary: {'f0_mean': 132.4, 'jitter_local': 0.89},
        trace: {'inference_ms': 19.0},
        name: 'Test User',
        age: 55,
        notes: 'Follow-up needed',
      );
      final restored = CheckRecord.fromJson(original.toJson());
      expect(restored.id, original.id);
      expect(restored.riskScore, original.riskScore);
      expect(restored.riskLabel, original.riskLabel);
      expect(restored.name, original.name);
      expect(restored.age, original.age);
      expect(restored.notes, original.notes);
      expect(restored.modelScores['xgb'], 0.82);
      expect(restored.featureSummary['jitter_local'], 0.89);
    });
  });

  group('CheckRecord copyWith', () {
    final base = CheckRecord(
      id: 'cw-1',
      timestamp: DateTime.utc(2026, 4, 11),
      source: 'analysis',
      audioFilePath: '/path/to/audio.wav',
      riskLabel: 'low',
      riskScore: 0.2,
      modelName: 'ensemble_v1',
      modelScores: {},
      featureSummary: {},
      trace: {},
      name: 'Alice',
      age: 60,
      notes: 'Initial check',
    );

    test('updates name only', () {
      final updated = base.copyWith(name: 'Bob');
      expect(updated.name, 'Bob');
      expect(updated.age, 60);
      expect(updated.notes, 'Initial check');
    });

    test('clears name with flag', () {
      final updated = base.copyWith(clearName: true);
      expect(updated.name, isNull);
      expect(updated.age, 60);
    });

    test('clears age with flag', () {
      final updated = base.copyWith(clearAge: true);
      expect(updated.age, isNull);
      expect(updated.name, 'Alice');
    });

    test('clears notes with flag', () {
      final updated = base.copyWith(clearNotes: true);
      expect(updated.notes, isNull);
    });

    test('preserves immutability', () {
      final updated = base.copyWith(name: 'Changed');
      expect(base.name, 'Alice');
      expect(updated.name, 'Changed');
    });

    test('updates audioFilePath', () {
      final updated = base.copyWith(audioFilePath: '/new/path.wav');
      expect(updated.audioFilePath, '/new/path.wav');
      expect(base.audioFilePath, '/path/to/audio.wav');
    });
  });
}
