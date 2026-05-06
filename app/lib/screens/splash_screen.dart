import 'dart:async';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../navigation/app_router.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// Splash screen (~2.5 s visible, then navigates based on first-launch state).
///
/// The tagline rotates through all supported languages to immediately showcase
/// multilingual capability — a key hackathon judging criterion.
///
/// Reads [SharedPreferences] key 'onboarding_completed':
///   • false / missing → onboarding
///   • true            → home
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _fadeController;
  late final Animation<double> _opacity;

  int _taglineIndex = 0;
  Timer? _taglineTimer;

  /// Tagline in each supported language — cycles on splash to show breadth.
  static const _taglines = [
    'hear what matters',
    'ascolta ciò che conta',
    '聆听重要之声',
    'escucha lo que importa',
    'écoutez ce qui compte',
  ];

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
    _opacity = CurvedAnimation(parent: _fadeController, curve: Curves.easeIn);
    _fadeController.forward();

    // Start tagline rotation after initial fade-in completes.
    Future.delayed(const Duration(milliseconds: 600), _startTaglineRotation);
    _navigate();
  }

  void _startTaglineRotation() {
    _taglineTimer = Timer.periodic(const Duration(milliseconds: 600), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      setState(() => _taglineIndex = (_taglineIndex + 1) % _taglines.length);
    });
  }

  Future<void> _navigate() async {
    // Extended to ~2.5 s so the tagline cycles through 3-4 languages.
    await Future<void>.delayed(const Duration(milliseconds: 2500));
    if (!mounted) return;

    final prefs = await SharedPreferences.getInstance();
    final onboardingDone = prefs.getBool('onboarding_completed') ?? false;

    if (!mounted) return;
    await Navigator.of(context).pushReplacementNamed(
      onboardingDone ? AppRoutes.home : AppRoutes.onboarding,
    );
  }

  @override
  void dispose() {
    _taglineTimer?.cancel();
    _fadeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: FadeTransition(
        opacity: _opacity,
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Logo mark — waveform in a rounded square
              Container(
                width: 76,
                height: 76,
                decoration: BoxDecoration(
                  color: AppColors.primaryContainer,
                  borderRadius: BorderRadius.circular(22),
                ),
                child: const Icon(
                  Icons.graphic_eq_rounded,
                  size: 42,
                  color: AppColors.primary,
                ),
              ),
              const SizedBox(height: 24),
              const Text('CogniTrace', style: AppTextStyles.displayMedium),
              const SizedBox(height: 8),
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 300),
                child: Text(
                  _taglines[_taglineIndex],
                  key: ValueKey(_taglineIndex),
                  style: AppTextStyles.tagline,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
