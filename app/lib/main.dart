import 'dart:async';

import 'package:flutter/material.dart';

import 'navigation/app_router.dart';
import 'services/gemma_download_manager.dart';
import 'theme/app_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Start Gemma download eagerly — runs in background while user navigates.
  unawaited(GemmaDownloadManager.instance.ensureStarted());
  runApp(const CogniTraceApp());
}

class CogniTraceApp extends StatelessWidget {
  const CogniTraceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CogniTrace',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      initialRoute: AppRoutes.splash,
      onGenerateRoute: onGenerateRoute,
    );
  }
}
