import 'dart:convert';

import '../navigation/app_router.dart';

class CheckRecord {
  const CheckRecord({
    required this.id,
    required this.timestamp,
    required this.source,
    required this.audioFilePath,
    required this.riskLabel,
    required this.riskScore,
    required this.modelName,
    required this.modelScores,
    required this.featureSummary,
    required this.trace,
    this.featureError,
    this.name,
    this.age,
    this.notes,
    this.taskSampleLengths,
  });

  final String id;
  final DateTime timestamp;
  final String source;
  final String audioFilePath;
  final String riskLabel;
  final double riskScore;
  final String modelName;
  final Map<String, double> modelScores;
  final Map<String, double> featureSummary;
  final Map<String, double> trace;
  final String? featureError;
  final String? name;
  final int? age;
  final String? notes;
  final List<int>? taskSampleLengths;

  CheckRecord copyWith({
    String? audioFilePath,
    String? name,
    int? age,
    String? notes,
    bool clearName = false,
    bool clearAge = false,
    bool clearNotes = false,
  }) {
    return CheckRecord(
      id: id,
      timestamp: timestamp,
      source: source,
      audioFilePath: audioFilePath ?? this.audioFilePath,
      riskLabel: riskLabel,
      riskScore: riskScore,
      modelName: modelName,
      modelScores: modelScores,
      featureSummary: featureSummary,
      trace: trace,
      featureError: featureError,
      name: clearName ? null : (name ?? this.name),
      age: clearAge ? null : (age ?? this.age),
      notes: clearNotes ? null : (notes ?? this.notes),
      taskSampleLengths: taskSampleLengths,
    );
  }

  factory CheckRecord.fromResultsArguments({
    required String id,
    required DateTime timestamp,
    required String source,
    required String audioFilePath,
    required ResultsArguments arguments,
  }) {
    return CheckRecord(
      id: id,
      timestamp: timestamp,
      source: source,
      audioFilePath: audioFilePath,
      riskLabel: arguments.riskLabel ?? 'low',
      riskScore: arguments.riskScore ?? 0,
      modelName: arguments.modelName ?? 'unknown',
      modelScores: arguments.modelScores ?? const {},
      featureSummary: arguments.featureSummary ?? const {},
      trace: arguments.trace ?? const {},
      featureError: arguments.featureError,
      taskSampleLengths: arguments.taskSampleLengths,
    );
  }

  factory CheckRecord.fromJson(Map<String, dynamic> json) {
    Map<String, double> parseMap(Object? raw) {
      final map = (raw as Map<String, dynamic>? ?? const <String, dynamic>{});
      return map.map((key, value) => MapEntry(key, (value as num).toDouble()));
    }

    return CheckRecord(
      id: json['id'] as String,
      timestamp: DateTime.parse(json['timestamp'] as String),
      source: json['source'] as String,
      audioFilePath: json['audioFilePath'] as String? ?? '',
      riskLabel: json['riskLabel'] as String,
      riskScore: (json['riskScore'] as num).toDouble(),
      modelName: json['modelName'] as String,
      modelScores: parseMap(json['modelScores']),
      featureSummary: parseMap(json['featureSummary']),
      trace: parseMap(json['trace']),
      featureError: json['featureError'] as String?,
      name: json['name'] as String?,
      age: json['age'] as int?,
      notes: json['notes'] as String?,
      taskSampleLengths: (json['taskSampleLengths'] as List<dynamic>?)
          ?.map((v) => (v as num).toInt())
          .toList(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'timestamp': timestamp.toIso8601String(),
      'source': source,
      'audioFilePath': audioFilePath,
      'riskLabel': riskLabel,
      'riskScore': riskScore,
      'modelName': modelName,
      'modelScores': modelScores,
      'featureSummary': featureSummary,
      'trace': trace,
      'featureError': featureError,
      if (name != null) 'name': name,
      if (age != null) 'age': age,
      if (notes != null) 'notes': notes,
      if (taskSampleLengths != null) 'taskSampleLengths': taskSampleLengths,
    };
  }

  ResultsArguments toResultsArguments() {
    return ResultsArguments(
      source: source,
      audioFilePath: audioFilePath,
      shouldAutoSave: false,
      riskScore: riskScore,
      sampleRate: null,
      recordedPcm: null,
      referenceWavBytes: null,
      riskLabel: riskLabel,
      modelName: modelName,
      modelScores: modelScores,
      trace: trace,
      featureSummary: featureSummary,
      featureError: featureError,
      taskSampleLengths: taskSampleLengths,
    );
  }

  static String encodeList(List<CheckRecord> records) {
    return jsonEncode(records.map((record) => record.toJson()).toList());
  }

  static List<CheckRecord> decodeList(String raw) {
    final decoded = jsonDecode(raw) as List<dynamic>;
    return decoded
        .map((entry) => CheckRecord.fromJson(entry as Map<String, dynamic>))
        .toList();
  }
}
