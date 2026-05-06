import 'package:shared_preferences/shared_preferences.dart';

/// Persists the user's preferred assessment language.
///
/// All Gemma-generated content and static UI strings render in this language.
/// Uses SharedPreferences for persistence across app launches.
class LanguagePreferenceService {
  static const _key = 'selected_language';
  static const defaultLanguage = 'English';

  /// Supported languages: display name (shown in UI) → Gemma prompt name.
  static const languages = <({String display, String gemma})>[
    (display: 'English', gemma: 'English'),
    (display: 'Italiano', gemma: 'Italian'),
    (display: '中文', gemma: 'Chinese'),
    (display: 'Español', gemma: 'Spanish'),
    (display: 'Français', gemma: 'French'),
  ];

  static List<String> get displayNames =>
      languages.map((l) => l.display).toList();

  /// Maps a UI display name to the name Gemma expects in its prompt.
  static String gemmaNameFor(String displayName) {
    return languages
            .where((l) => l.display == displayName)
            .firstOrNull
            ?.gemma ??
        displayName;
  }

  static Future<String> load() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_key) ?? defaultLanguage;
  }

  static Future<void> save(String language) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, language);
  }
}
