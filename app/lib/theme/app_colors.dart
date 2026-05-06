import 'package:flutter/material.dart';

/// CogniTrace design-system color tokens.
///
/// Palette rationale (from functional spec):
///   Background  #FAFAF8  — clinical warmth, not cold white
///   Primary     #0D7377  — medical trust (deep teal)
///   Risk low    #3D8B37  — reassuring without celebrating
///   Risk mod    #C4841D  — attention without alarm
///   Risk high   #9B2335  — serious without screaming red
abstract final class AppColors {
  // ── Backgrounds ─────────────────────────────────────────────────────────────
  static const Color background = Color(0xFFFAFAF8);
  static const Color surface = Color(0xFFFFFFFF);
  static const Color surfaceElevated = Color(0xFFF5F5F2);

  // ── Primary action — deep teal ───────────────────────────────────────────
  static const Color primary = Color(0xFF0D7377);
  static const Color primaryDark = Color(0xFF095C60);
  static const Color primaryContainer = Color(0xFFCDE8E9);
  static const Color onPrimary = Color(0xFFFFFFFF);

  // ── Risk indicators ──────────────────────────────────────────────────────
  static const Color riskLow = Color(0xFF3D8B37);
  static const Color riskModerate = Color(0xFFC4841D);
  static const Color riskElevated = Color(0xFF9B2335);

  // Risk container backgrounds (very desaturated tints)
  static const Color riskLowContainer = Color(0xFFEEF7ED);
  static const Color riskModerateContainer = Color(0xFFFBF3E8);
  static const Color riskElevatedContainer = Color(0xFFF7ECED);

  // ── Text ─────────────────────────────────────────────────────────────────
  static const Color textPrimary = Color(0xFF1C1C1E);
  static const Color textSecondary = Color(0xFF636366);
  static const Color textTertiary = Color(0xFFAEAEB2);
  static const Color textOnPrimary = Color(0xFFFFFFFF);

  // ── Structural ───────────────────────────────────────────────────────────
  static const Color border = Color(0xFFE5E5EA);
  static const Color divider = Color(0xFFEAEAE6);

  // Transparent overlays
  static const Color shadowSoft = Color(0x0A000000);
  static const Color shadowMedium = Color(0x14000000);
}
