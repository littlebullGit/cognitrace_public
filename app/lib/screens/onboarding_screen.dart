import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../l10n/app_strings.dart';
import '../navigation/app_router.dart';
import '../services/language_preference_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// 4-card onboarding flow shown on first launch only.
///
/// Cards advance manually (swipe or tap "Next").
/// Gemma downloads separately in the background and is only required for the
/// result interpretation step, so onboarding never blocks voice recording.
///
/// On completion: sets SharedPreferences 'onboarding_completed' = true
/// and navigates to [AppRoutes.home].
class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _pageController = PageController();
  int _currentPage = 0;
  String _language = LanguagePreferenceService.defaultLanguage;

  @override
  void initState() {
    super.initState();
    unawaited(_loadLanguage());
  }

  Future<void> _loadLanguage() async {
    final lang = await LanguagePreferenceService.load();
    if (mounted) setState(() => _language = lang);
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  void _goToNextPage() {
    if (_currentPage < 3) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 350),
        curve: Curves.easeInOut,
      );
    }
  }

  Future<void> _beginVoiceCheck() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('onboarding_completed', true);
    if (!mounted) return;
    await Navigator.of(context).pushReplacementNamed(AppRoutes.home);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: PageView(
                controller: _pageController,
                onPageChanged: (i) => setState(() => _currentPage = i),
                children: [
                  _Card1(language: _language),
                  _Card2(language: _language),
                  _Card3(language: _language),
                  _Card4(language: _language, onBegin: _beginVoiceCheck),
                ],
              ),
            ),
            _BottomBar(
              language: _language,
              currentPage: _currentPage,
              onNext: _goToNextPage,
            ),
          ],
        ),
      ),
    );
  }
}

// ── Card 1: The Hook ─────────────────────────────────────────────────────────

class _Card1 extends StatelessWidget {
  final String language;

  const _Card1({required this.language});

  @override
  Widget build(BuildContext context) {
    return _CardScroll(
      children: [
        const SizedBox(height: 28),
        const _WaveformIllustration(),
        const SizedBox(height: 36),
        Text(
          AppStrings.get('onboard_hook_title', language),
          style: AppTextStyles.displayMedium,
        ),
        const SizedBox(height: 20),
        Text(
          AppStrings.get('onboard_hook_body', language),
          style: AppTextStyles.bodyLarge,
        ),
      ],
    );
  }
}

// ── Card 2: How It Works ─────────────────────────────────────────────────────

class _Card2 extends StatelessWidget {
  final String language;

  const _Card2({required this.language});

  @override
  Widget build(BuildContext context) {
    return _CardScroll(
      children: [
        const SizedBox(height: 28),
        Text(
          AppStrings.get('onboard_how_title', language),
          style: AppTextStyles.displayMedium,
        ),
        const SizedBox(height: 36),
        Row(
          children: [
            _StepBadge(
              icon: Icons.mic_none_rounded,
              label: AppStrings.get('onboard_step_speak', language),
            ),
            const SizedBox(width: 8),
            const Icon(
              Icons.arrow_forward_rounded,
              size: 18,
              color: AppColors.textTertiary,
            ),
            const SizedBox(width: 8),
            _StepBadge(
              icon: Icons.analytics_outlined,
              label: AppStrings.get('onboard_step_analyze', language),
            ),
            const SizedBox(width: 8),
            const Icon(
              Icons.arrow_forward_rounded,
              size: 18,
              color: AppColors.textTertiary,
            ),
            const SizedBox(width: 8),
            _StepBadge(
              icon: Icons.chat_bubble_outline_rounded,
              label: AppStrings.get('onboard_step_explain', language),
            ),
          ],
        ),
        const SizedBox(height: 36),
        Text(
          AppStrings.get('onboard_how_body', language),
          style: AppTextStyles.bodyLarge,
        ),
      ],
    );
  }
}

// ── Card 3: Any Language ─────────────────────────────────────────────────────

class _Card3 extends StatelessWidget {
  final String language;

  const _Card3({required this.language});

  @override
  Widget build(BuildContext context) {
    final languages = ['English', 'Italiano', '中文', 'Español', 'Français'];

    return _CardScroll(
      children: [
        const SizedBox(height: 28),
        Text(
          AppStrings.get('onboard_lang_title', language),
          style: AppTextStyles.displayMedium,
        ),
        const SizedBox(height: 20),
        Text(
          AppStrings.get('onboard_lang_body', language),
          style: AppTextStyles.bodyLarge,
        ),
        const SizedBox(height: 28),
        ...languages.map(
          (lang) => Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Row(
              children: [
                Container(
                  width: 5,
                  height: 5,
                  decoration: const BoxDecoration(
                    color: AppColors.primary,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 14),
                Text(lang, style: AppTextStyles.bodyMedium),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

// ── Card 4: Ready ────────────────────────────────────────────────────────────

class _Card4 extends StatelessWidget {
  final String language;
  final VoidCallback onBegin;

  const _Card4({required this.language, required this.onBegin});

  @override
  Widget build(BuildContext context) {
    return _CardScroll(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const SizedBox(height: 48),
        Center(
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 500),
            transitionBuilder: (child, anim) => ScaleTransition(
              scale: anim,
              child: FadeTransition(opacity: anim, child: child),
            ),
            child: Container(
              key: const ValueKey('check'),
              width: 76,
              height: 76,
              decoration: const BoxDecoration(
                color: AppColors.riskLowContainer,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.check_rounded,
                size: 38,
                color: AppColors.riskLow,
              ),
            ),
          ),
        ),
        const SizedBox(height: 32),
        Text(
          AppStrings.get('onboard_ready', language),
          style: AppTextStyles.displayMedium,
        ),
        const SizedBox(height: 20),
        Text(
          AppStrings.get('onboard_disclaimer', language),
          style: AppTextStyles.bodyLarge,
        ),
        const SizedBox(height: 40),
        ElevatedButton(
          onPressed: onBegin,
          child: Text(AppStrings.get('onboard_begin', language)),
        ),
        const SizedBox(height: 16),
      ],
    );
  }
}

// ── Bottom bar ───────────────────────────────────────────────────────────────

class _BottomBar extends StatelessWidget {
  final String language;
  final int currentPage;
  final VoidCallback onNext;

  const _BottomBar({
    required this.language,
    required this.currentPage,
    required this.onNext,
  });

  @override
  Widget build(BuildContext context) {
    final isLastCard = currentPage == 3;
    final pageLabel = AppStrings.get(
      'onboard_x_of_4',
      language,
    ).replaceAll('{n}', '${currentPage + 1}');
    final pageProgress = (currentPage + 1) / 4;

    return Container(
      padding: const EdgeInsets.fromLTRB(24, 12, 24, 32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(2),
            child: LinearProgressIndicator(
              value: pageProgress,
              minHeight: 3,
              color: AppColors.primary,
              backgroundColor: AppColors.border,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: List.generate(4, (i) {
                  final active = i == currentPage;
                  return AnimatedContainer(
                    duration: const Duration(milliseconds: 250),
                    curve: Curves.easeInOut,
                    margin: const EdgeInsets.only(right: 6),
                    width: active ? 22.0 : 6.0,
                    height: 6,
                    decoration: BoxDecoration(
                      color: active ? AppColors.primary : AppColors.border,
                      borderRadius: BorderRadius.circular(3),
                    ),
                  );
                }),
              ),
              if (!isLastCard)
                Row(
                  children: [
                    Text(pageLabel, style: AppTextStyles.caption),
                    const SizedBox(width: 14),
                    TextButton(
                      onPressed: onNext,
                      style: TextButton.styleFrom(
                        backgroundColor: AppColors.primaryContainer,
                        foregroundColor: AppColors.primary,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 10,
                        ),
                        minimumSize: const Size(44, 44),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(22),
                        ),
                      ),
                      child: Text(AppStrings.get('onboard_next', language)),
                    ),
                  ],
                )
              else
                Text(pageLabel, style: AppTextStyles.caption),
            ],
          ),
        ],
      ),
    );
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

class _CardScroll extends StatelessWidget {
  final List<Widget> children;
  final MainAxisAlignment mainAxisAlignment;

  const _CardScroll({
    required this.children,
    this.mainAxisAlignment = MainAxisAlignment.start,
  });

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 28),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: mainAxisAlignment,
        children: children,
      ),
    );
  }
}

class _StepBadge extends StatelessWidget {
  final IconData icon;
  final String label;

  const _StepBadge({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        children: [
          Container(
            height: 56,
            decoration: BoxDecoration(
              color: AppColors.primaryContainer,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(icon, size: 26, color: AppColors.primary),
          ),
          const SizedBox(height: 8),
          Text(
            label,
            style: AppTextStyles.captionStrong,
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}

// ── Waveform illustration (Card 1) ────────────────────────────────────────────

class _WaveformIllustration extends StatefulWidget {
  const _WaveformIllustration();

  @override
  State<_WaveformIllustration> createState() => _WaveformIllustrationState();
}

class _WaveformIllustrationState extends State<_WaveformIllustration>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 72,
      child: AnimatedBuilder(
        animation: _ctrl,
        builder: (_, child) => CustomPaint(
          painter: _WaveformPainter(phase: _ctrl.value),
          size: const Size(double.infinity, 72),
        ),
      ),
    );
  }
}

class _WaveformPainter extends CustomPainter {
  final double phase;

  const _WaveformPainter({required this.phase});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AppColors.primary
          .withAlpha(153) // ~60 % opacity
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    const barCount = 34;
    final slotW = size.width / barCount;
    final maxH = size.height * 0.85;
    final cy = size.height / 2;

    for (var i = 0; i < barCount; i++) {
      final x = (i + 0.5) * slotW;
      final t = (i / barCount + phase) * math.pi * 2;
      final raw =
          math.sin(t) * 0.45 +
          math.sin(t * 2.4 + 1.1) * 0.30 +
          math.sin(t * 0.75 + 2.0) * 0.25;
      final h = raw.abs().clamp(0.05, 1.0) * maxH;
      canvas.drawLine(Offset(x, cy - h / 2), Offset(x, cy + h / 2), paint);
    }
  }

  @override
  bool shouldRepaint(_WaveformPainter old) => old.phase != phase;
}
