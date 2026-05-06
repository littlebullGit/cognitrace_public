import 'package:flutter_test/flutter_test.dart';
import 'package:cognitrace/l10n/app_strings.dart';

void main() {
  group('AppStrings', () {
    test('English returns value for known key', () {
      final val = AppStrings.get('app_title', 'English');
      expect(val, isNotEmpty);
      expect(val, contains('CogniTrace'));
    });

    test('all 5 languages have app_title', () {
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        final val = AppStrings.get('app_title', lang);
        expect(val, isNotEmpty, reason: '$lang missing app_title');
      }
    });

    test('missing key returns the key itself', () {
      final val = AppStrings.get('nonexistent_key_xyz', 'English');
      expect(val, 'nonexistent_key_xyz');
    });

    test('unknown language falls back to English', () {
      final val = AppStrings.get('app_title', 'Klingon');
      final english = AppStrings.get('app_title', 'English');
      expect(val, english);
    });

    test('all languages have disclaimer with 50+', () {
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        final val = AppStrings.get('disclaimer_short', lang);
        expect(
          val,
          contains('50'),
          reason: '$lang disclaimer_short missing 50+',
        );
      }
    });

    test('all languages have age_guidance', () {
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        final val = AppStrings.get('age_guidance', lang);
        expect(val, isNotEmpty, reason: '$lang missing age_guidance');
        expect(val, contains('50'), reason: '$lang age_guidance missing 50');
      }
    });

    test('all languages have chat keys', () {
      const chatKeys = [
        'ask_gemma',
        'chat_placeholder',
        'chat_disclaimer',
        'chat_thinking',
      ];
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        for (final key in chatKeys) {
          final val = AppStrings.get(key, lang);
          expect(val, isNot(key), reason: '$lang missing $key');
        }
      }
    });

    test('all languages have notes keys', () {
      const notesKeys = [
        'edit_notes',
        'name_label',
        'age_label',
        'notes_label',
        'save',
      ];
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        for (final key in notesKeys) {
          final val = AppStrings.get(key, lang);
          expect(val, isNot(key), reason: '$lang missing $key');
        }
      }
    });

    test('all languages have doctor guide keys', () {
      const doctorKeys = [
        'prepare_doctor_visit',
        'doctor_discussion_guide',
        'generating_doctor_guide',
        'share_guide',
        'visit_reason',
        'result_summary',
        'questions_to_ask',
        'context_to_share',
        'caveats',
      ];
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        for (final key in doctorKeys) {
          final val = AppStrings.get(key, lang);
          expect(val, isNot(key), reason: '$lang missing $key');
        }
      }
    });

    test('all languages have privacy data deletion keys', () {
      const privacyKeys = [
        'privacy_desc',
        'delete_saved_data',
        'delete_saved_data_body',
        'delete_saved_data_confirm_title',
        'delete_saved_data_confirm_body',
        'delete_saved_data_confirm_action',
        'delete_saved_data_done',
        'delete_saved_data_failed',
      ];
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        for (final key in privacyKeys) {
          final val = AppStrings.get(key, lang);
          expect(val, isNot(key), reason: '$lang missing $key');
        }
      }
    });

    test('all languages have doctor guide context keys', () {
      const doctorContextKeys = [
        'doctor_guide_personalize_body',
        'doctor_guide_reason_label',
        'doctor_guide_reason_hint',
        'doctor_guide_changes_label',
        'doctor_guide_changes_hint',
        'doctor_guide_recording_factors',
        'doctor_factor_tired',
        'doctor_factor_sick',
        'doctor_factor_noisy_room',
        'generate_guide',
      ];
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        for (final key in doctorContextKeys) {
          final val = AppStrings.get(key, lang);
          expect(val, isNot(key), reason: '$lang missing $key');
        }
      }
    });

    test('all languages have education keys', () {
      const educationKeys = [
        'learn_about_result',
        'for_you_right_now',
        'loading_education',
        'education_unavailable',
        'baseline_saved_hint',
      ];
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        for (final key in educationKeys) {
          final val = AppStrings.get(key, lang);
          expect(val, isNot(key), reason: '$lang missing $key');
        }
      }
    });

    test('all languages have results storytelling keys', () {
      const resultsStoryKeys = [
        'voice_tells_us',
        'same_result_your_language_title',
        'same_result_your_language_body',
        'raw_analysis_title',
        'raw_analysis_body',
      ];
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        for (final key in resultsStoryKeys) {
          final val = AppStrings.get(key, lang);
          expect(val, isNot(key), reason: '$lang missing $key');
        }
      }
    });

    test('English has all risk level strings', () {
      expect(AppStrings.get('risk_low', 'English'), isNotEmpty);
      expect(AppStrings.get('risk_moderate', 'English'), isNotEmpty);
      expect(AppStrings.get('risk_elevated', 'English'), isNotEmpty);
    });

    test('all languages have risk level strings', () {
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        expect(
          AppStrings.get('risk_low', lang),
          isNotEmpty,
          reason: '$lang missing risk_low',
        );
        expect(
          AppStrings.get('risk_moderate', lang),
          isNotEmpty,
          reason: '$lang missing risk_moderate',
        );
        expect(
          AppStrings.get('risk_elevated', lang),
          isNotEmpty,
          reason: '$lang missing risk_elevated',
        );
      }
    });

    test('all languages have practice conversation keys', () {
      const practiceKeys = [
        'practice_conversation',
        'practice_title',
        'end_practice',
        'practice_summary_title',
        'practice_generating',
        'practice_ended',
        'practice_end_prompt',
      ];
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        for (final key in practiceKeys) {
          final val = AppStrings.get(key, lang);
          expect(val, isNot(key), reason: '$lang missing $key');
        }
      }
    });

    test('all languages have task 3 voice capture guidance keys', () {
      const captureKeys = [
        'voice_capture_label',
        'voice_capture_keep_speaking',
        'voice_capture_hold_until_timer',
        'voice_capture_ready',
      ];
      for (final lang in ['English', 'Italiano', '中文', 'Español', 'Français']) {
        for (final key in captureKeys) {
          final val = AppStrings.get(key, lang);
          expect(val, isNot(key), reason: '$lang missing $key');
        }
      }
    });

    test('English task 3 guidance is actionable and normal-voice oriented', () {
      expect(
        AppStrings.get('speech_detail', 'English').toLowerCase(),
        contains('normal voice'),
      );
      expect(AppStrings.get('voice_capture_label', 'English'), 'Speech signal');
      expect(
        AppStrings.get('voice_capture_keep_speaking', 'English').toLowerCase(),
        allOf(contains('phone'), contains('normal voice')),
      );
      expect(
        AppStrings.get('voice_capture_ready', 'English'),
        'Good speech sample. You can stop now.',
      );
    });
  });
}
