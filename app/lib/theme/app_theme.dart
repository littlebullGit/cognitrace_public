import 'package:flutter/material.dart';

import 'app_colors.dart';
import 'app_text_styles.dart';

/// CogniTrace ThemeData.
///
/// Medical-grade aesthetic: calm authority, generous typography,
/// muted teal primary. Think Apple Health meets Mayo Clinic.
abstract final class AppTheme {
  static ThemeData get light => ThemeData(
    useMaterial3: true,
    colorScheme: const ColorScheme.light(
      primary: AppColors.primary,
      onPrimary: AppColors.onPrimary,
      primaryContainer: AppColors.primaryContainer,
      secondary: AppColors.riskLow,
      surface: AppColors.surface,
      error: AppColors.riskElevated,
      onSurface: AppColors.textPrimary,
      outline: AppColors.border,
      outlineVariant: AppColors.divider,
    ),
    scaffoldBackgroundColor: AppColors.background,

    // ── AppBar ────────────────────────────────────────────────────────
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.background,
      foregroundColor: AppColors.textPrimary,
      elevation: 0,
      scrolledUnderElevation: 0.5,
      shadowColor: AppColors.shadowSoft,
      surfaceTintColor: Colors.transparent,
      titleTextStyle: AppTextStyles.headingSmall,
      iconTheme: IconThemeData(color: AppColors.primary, size: 24),
    ),

    // ── Elevated Button (primary CTA) ─────────────────────────────────
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.primary,
        foregroundColor: AppColors.onPrimary,
        disabledBackgroundColor: AppColors.border,
        disabledForegroundColor: AppColors.textTertiary,
        elevation: 0,
        shadowColor: Colors.transparent,
        minimumSize: const Size(double.infinity, 56),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        textStyle: AppTextStyles.buttonLarge,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      ),
    ),

    // ── Outlined Button ───────────────────────────────────────────────
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: AppColors.primary,
        side: const BorderSide(color: AppColors.primary, width: 1.5),
        minimumSize: const Size(double.infinity, 52),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        textStyle: AppTextStyles.buttonLarge,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      ),
    ),

    // ── Text Button ───────────────────────────────────────────────────
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: AppColors.primary,
        minimumSize: const Size(44, 44),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        textStyle: AppTextStyles.buttonLarge,
      ),
    ),

    // ── Card ──────────────────────────────────────────────────────────
    cardTheme: CardThemeData(
      color: AppColors.surface,
      elevation: 0,
      shadowColor: Colors.transparent,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: AppColors.border),
      ),
      margin: const EdgeInsets.symmetric(vertical: 6),
    ),

    // ── Divider ───────────────────────────────────────────────────────
    dividerTheme: const DividerThemeData(
      color: AppColors.divider,
      thickness: 1,
      space: 1,
    ),

    // ── Input Decoration ──────────────────────────────────────────────
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AppColors.surfaceElevated,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppColors.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppColors.border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppColors.primary, width: 1.5),
      ),
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      hintStyle: AppTextStyles.bodyMediumSecondary,
    ),

    // ── List Tile ─────────────────────────────────────────────────────
    listTileTheme: const ListTileThemeData(
      contentPadding: EdgeInsets.symmetric(horizontal: 20, vertical: 2),
      minLeadingWidth: 0,
      minTileHeight: 52,
    ),

    // ── Progress Indicator ────────────────────────────────────────────
    progressIndicatorTheme: const ProgressIndicatorThemeData(
      color: AppColors.primary,
      linearTrackColor: AppColors.border,
    ),

    // ── Text Theme ────────────────────────────────────────────────────
    textTheme: const TextTheme(
      displayLarge: AppTextStyles.displayLarge,
      displayMedium: AppTextStyles.displayMedium,
      headlineLarge: AppTextStyles.headingLarge,
      headlineMedium: AppTextStyles.headingMedium,
      headlineSmall: AppTextStyles.headingSmall,
      bodyLarge: AppTextStyles.bodyLarge,
      bodyMedium: AppTextStyles.bodyMedium,
      bodySmall: AppTextStyles.caption,
      labelLarge: AppTextStyles.labelLarge,
      labelMedium: AppTextStyles.labelMedium,
      labelSmall: AppTextStyles.captionStrong,
    ),
  );
}
