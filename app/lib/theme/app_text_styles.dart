import 'package:flutter/material.dart';

import 'app_colors.dart';

/// CogniTrace typography scale.
///
/// Rules (from functional spec):
///   • Minimum 16pt for all body text — older users are primary audience
///   • 14pt only for captions / fine print
///   • No fontFamily override: iOS system font resolves to SF Pro automatically
abstract final class AppTextStyles {
  // ── Display ───────────────────────────────────────────────────────────────
  static const TextStyle displayLarge = TextStyle(
    fontSize: 32,
    fontWeight: FontWeight.w700,
    letterSpacing: -0.8,
    height: 1.15,
    color: AppColors.textPrimary,
  );

  static const TextStyle displayMedium = TextStyle(
    fontSize: 28,
    fontWeight: FontWeight.w600,
    letterSpacing: -0.5,
    height: 1.2,
    color: AppColors.textPrimary,
  );

  // ── Headings ──────────────────────────────────────────────────────────────
  static const TextStyle headingLarge = TextStyle(
    fontSize: 24,
    fontWeight: FontWeight.w600,
    letterSpacing: -0.3,
    height: 1.25,
    color: AppColors.textPrimary,
  );

  static const TextStyle headingMedium = TextStyle(
    fontSize: 20,
    fontWeight: FontWeight.w600,
    letterSpacing: -0.2,
    height: 1.3,
    color: AppColors.textPrimary,
  );

  static const TextStyle headingSmall = TextStyle(
    fontSize: 17,
    fontWeight: FontWeight.w600,
    letterSpacing: -0.1,
    height: 1.35,
    color: AppColors.textPrimary,
  );

  // ── Body — 16pt minimum ───────────────────────────────────────────────────
  static const TextStyle bodyLarge = TextStyle(
    fontSize: 18,
    fontWeight: FontWeight.w400,
    letterSpacing: 0,
    height: 1.55,
    color: AppColors.textPrimary,
  );

  static const TextStyle bodyMedium = TextStyle(
    fontSize: 16,
    fontWeight: FontWeight.w400,
    letterSpacing: 0,
    height: 1.5,
    color: AppColors.textPrimary,
  );

  static const TextStyle bodyMediumSecondary = TextStyle(
    fontSize: 16,
    fontWeight: FontWeight.w400,
    letterSpacing: 0,
    height: 1.5,
    color: AppColors.textSecondary,
  );

  // ── Caption — 14pt only for fine print / labels ───────────────────────────
  static const TextStyle caption = TextStyle(
    fontSize: 14,
    fontWeight: FontWeight.w400,
    letterSpacing: 0.1,
    height: 1.45,
    color: AppColors.textSecondary,
  );

  static const TextStyle captionStrong = TextStyle(
    fontSize: 14,
    fontWeight: FontWeight.w500,
    letterSpacing: 0.1,
    height: 1.45,
    color: AppColors.textPrimary,
  );

  // ── Button ────────────────────────────────────────────────────────────────
  static const TextStyle buttonLarge = TextStyle(
    fontSize: 17,
    fontWeight: FontWeight.w500,
    letterSpacing: 0,
    height: 1.3,
  );

  // ── Label ─────────────────────────────────────────────────────────────────
  static const TextStyle labelLarge = TextStyle(
    fontSize: 16,
    fontWeight: FontWeight.w500,
    letterSpacing: 0.1,
    height: 1.4,
    color: AppColors.textPrimary,
  );

  static const TextStyle labelMedium = TextStyle(
    fontSize: 14,
    fontWeight: FontWeight.w500,
    letterSpacing: 0.2,
    height: 1.4,
    color: AppColors.textSecondary,
  );

  // ── Special ───────────────────────────────────────────────────────────────
  static const TextStyle tagline = TextStyle(
    fontSize: 18,
    fontWeight: FontWeight.w300,
    letterSpacing: 0.6,
    height: 1.4,
    color: AppColors.textSecondary,
  );

  static const TextStyle sectionHeader = TextStyle(
    fontSize: 13,
    fontWeight: FontWeight.w600,
    letterSpacing: 0.5,
    height: 1.4,
    color: AppColors.textSecondary,
  );
}
