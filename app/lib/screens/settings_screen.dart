import 'dart:async';

import 'package:flutter/material.dart';

import '../l10n/app_strings.dart';
import '../services/audio_archive_service.dart';
import '../services/check_history_service.dart';
import '../services/gemma_download_manager.dart';
import '../services/gemma_service.dart';
import '../services/language_preference_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// Settings screen — minimal, per functional spec §7.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  String _language = LanguagePreferenceService.defaultLanguage;
  bool _isDeletingSavedData = false;

  @override
  void initState() {
    super.initState();
    unawaited(_loadLanguage());
    GemmaDownloadManager.instance.addListener(_onGemmaUpdate);
  }

  @override
  void dispose() {
    GemmaDownloadManager.instance.removeListener(_onGemmaUpdate);
    super.dispose();
  }

  void _onGemmaUpdate() {
    if (mounted) setState(() {});
  }

  Future<void> _loadLanguage() async {
    final lang = await LanguagePreferenceService.load();
    if (!mounted) return;
    setState(() => _language = lang);
  }

  String _s(String key) => AppStrings.get(key, _language);

  @override
  Widget build(BuildContext context) {
    final mgr = GemmaDownloadManager.instance;
    final modelReady = mgr.state == GemmaModelState.ready;
    final downloading = mgr.state == GemmaModelState.downloading;
    final pct = (mgr.progress * 100).toStringAsFixed(0);

    return Scaffold(
      appBar: AppBar(
        title: Text(_s('settings')),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: SafeArea(
        child: ListView(
          children: [
            _SectionHeader(label: _s('assessment_language')),
            _SettingsCard(
              children: [
                ListTile(
                  title: Text(_language, style: AppTextStyles.bodyMedium),
                  trailing: const Icon(
                    Icons.keyboard_arrow_down_rounded,
                    color: AppColors.textSecondary,
                  ),
                  onTap: () => _showLanguagePicker(),
                ),
              ],
            ),
            _SectionHeader(label: _s('ai_model')),
            _SettingsCard(
              children: [
                ListTile(
                  title: Text(
                    _s('gemma_4_e2b'),
                    style: AppTextStyles.bodyMedium,
                  ),
                  subtitle: Text(
                    downloading
                        ? _s('downloading_gemma').replaceAll('{pct}', pct)
                        : '2.7 GB',
                    style: AppTextStyles.caption,
                  ),
                  trailing: modelReady
                      ? const Icon(
                          Icons.check_rounded,
                          color: AppColors.riskLow,
                        )
                      : downloading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(
                          Icons.cloud_download_outlined,
                          color: AppColors.textSecondary,
                        ),
                ),
                const Divider(height: 1, indent: 16, endIndent: 16),
                ListTile(
                  title: Text(
                    _s('redownload_model'),
                    style: AppTextStyles.bodyMedium,
                  ),
                  trailing: const Icon(
                    Icons.download_outlined,
                    color: AppColors.primary,
                  ),
                  onTap: downloading
                      ? null
                      : () => unawaited(GemmaDownloadManager.instance.retry()),
                ),
              ],
            ),
            _SectionHeader(label: _s('about')),
            _SettingsCard(
              children: [
                const ListTile(
                  title: Text('CogniTrace', style: AppTextStyles.bodyMedium),
                  trailing: Text('v1.0', style: AppTextStyles.caption),
                ),
                const Divider(height: 1, indent: 16, endIndent: 16),
                ListTile(
                  title: Text(
                    _s('screening_disclaimer'),
                    style: AppTextStyles.bodyMediumSecondary,
                  ),
                ),
              ],
            ),
            _SectionHeader(label: _s('privacy')),
            _SettingsCard(
              children: [
                ListTile(
                  title: Text(
                    _s('privacy_desc'),
                    style: AppTextStyles.bodyMediumSecondary,
                  ),
                ),
                const Divider(height: 1, indent: 16, endIndent: 16),
                ListTile(
                  title: Text(
                    _s('delete_saved_data'),
                    style: AppTextStyles.bodyMedium.copyWith(
                      color: AppColors.riskElevated,
                    ),
                  ),
                  subtitle: Text(
                    _s('delete_saved_data_body'),
                    style: AppTextStyles.caption,
                  ),
                  trailing: _isDeletingSavedData
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(
                          Icons.delete_outline,
                          color: AppColors.riskElevated,
                        ),
                  onTap: _isDeletingSavedData
                      ? null
                      : () => unawaited(_confirmDeleteSavedData()),
                ),
                const Divider(height: 1, indent: 16, endIndent: 16),
                ListTile(
                  title: Text(
                    _s('open_source_licenses'),
                    style: AppTextStyles.bodyMedium,
                  ),
                  trailing: const Icon(
                    Icons.chevron_right_rounded,
                    color: AppColors.textSecondary,
                  ),
                  onTap: () => showLicensePage(context: context),
                ),
              ],
            ),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  Future<void> _confirmDeleteSavedData() async {
    final confirmed =
        await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: Text(_s('delete_saved_data_confirm_title')),
            content: Text(_s('delete_saved_data_confirm_body')),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: Text(_s('cancel')),
              ),
              TextButton(
                onPressed: () => Navigator.pop(ctx, true),
                child: Text(
                  _s('delete_saved_data_confirm_action'),
                  style: const TextStyle(color: AppColors.riskElevated),
                ),
              ),
            ],
          ),
        ) ??
        false;
    if (!confirmed || !mounted) return;

    setState(() => _isDeletingSavedData = true);
    try {
      await AudioArchiveService.deleteAllAudio();
      await CheckHistoryService.deleteAll();
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(_s('delete_saved_data_done'))));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            _s('delete_saved_data_failed').replaceAll('{error}', e.toString()),
          ),
        ),
      );
    } finally {
      if (mounted) setState(() => _isDeletingSavedData = false);
    }
  }

  void _showLanguagePicker() {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => Padding(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 40),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 36,
                height: 4,
                decoration: BoxDecoration(
                  color: AppColors.border,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 20),
            Text(_s('assessment_language'), style: AppTextStyles.headingMedium),
            const SizedBox(height: 12),
            ...LanguagePreferenceService.displayNames.map(
              (lang) => ListTile(
                contentPadding: EdgeInsets.zero,
                title: Text(lang, style: AppTextStyles.bodyMedium),
                trailing: _language == lang
                    ? const Icon(Icons.check_rounded, color: AppColors.primary)
                    : null,
                onTap: () {
                  setState(() => _language = lang);
                  unawaited(LanguagePreferenceService.save(lang));
                  Navigator.pop(ctx);
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Section helpers ───────────────────────────────────────────────────────────

class _SectionHeader extends StatelessWidget {
  final String label;

  const _SectionHeader({required this.label});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 28, 20, 8),
      child: Text(label.toUpperCase(), style: AppTextStyles.sectionHeader),
    );
  }
}

class _SettingsCard extends StatelessWidget {
  final List<Widget> children;

  const _SettingsCard({required this.children});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Container(
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.border),
        ),
        child: Column(children: children),
      ),
    );
  }
}
