import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../screens/analysis_screen.dart';
import '../screens/home_screen.dart';
import '../screens/onboarding_screen.dart';
import '../screens/record_screen.dart';
import '../screens/results_screen.dart';
import '../screens/settings_screen.dart';
import '../screens/splash_screen.dart';

class AnalysisArguments {
  const AnalysisArguments({
    required this.recordedPcm,
    required this.sampleRate,
    this.taskPcmList,
    this.taskSampleLengths,
    this.isReferenceSample = false,
    this.referenceWavBytes,
    this.sourceTag,
    this.shouldAutoSave = true,
  });

  final Float32List recordedPcm;
  final double sampleRate;
  final List<Float32List>? taskPcmList;
  final List<int>? taskSampleLengths;
  final bool isReferenceSample;
  final Uint8List? referenceWavBytes;
  final String? sourceTag;
  final bool shouldAutoSave;
}

class ResultsArguments {
  const ResultsArguments({
    this.source,
    this.audioFilePath,
    this.recordedPcm,
    this.referenceWavBytes,
    this.sampleRate,
    this.shouldAutoSave = true,
    this.riskScore,
    this.riskLabel,
    this.modelName,
    this.modelScores,
    this.trace,
    this.featureSummary,
    this.featureError,
    this.taskSampleLengths,
  });

  final String? source;
  final String? audioFilePath;
  final Float32List? recordedPcm;
  final Uint8List? referenceWavBytes;
  final double? sampleRate;
  final bool shouldAutoSave;
  final double? riskScore;
  final String? riskLabel;
  final String? modelName;
  final Map<String, double>? modelScores;
  final Map<String, double>? trace;
  final Map<String, double>? featureSummary;
  final String? featureError;
  final List<int>? taskSampleLengths;
}

/// Named route constants used throughout the app.
abstract final class AppRoutes {
  static const String splash = '/';
  static const String onboarding = '/onboarding';
  static const String home = '/home';
  static const String record = '/record';
  static const String analysis = '/analysis';
  static const String results = '/results';
  static const String settings = '/settings';
}

/// Route factory — registered as [MaterialApp.onGenerateRoute].
Route<dynamic> onGenerateRoute(RouteSettings routeSettings) {
  final Widget page;

  switch (routeSettings.name) {
    case AppRoutes.splash:
      page = const SplashScreen();
    case AppRoutes.onboarding:
      page = const OnboardingScreen();
    case AppRoutes.home:
      page = const HomeScreen();
    case AppRoutes.record:
      page = const RecordScreen();
    case AppRoutes.analysis:
      final args = routeSettings.arguments as AnalysisArguments?;
      page = AnalysisScreen(arguments: args);
    case AppRoutes.results:
      final args = routeSettings.arguments as ResultsArguments?;
      page = ResultsScreen(arguments: args);
    case AppRoutes.settings:
      page = const SettingsScreen();
    default:
      page = const SplashScreen();
  }

  return MaterialPageRoute<void>(builder: (_) => page, settings: routeSettings);
}
