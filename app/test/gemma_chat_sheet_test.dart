import 'package:cognitrace/utils/markdown_utils.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('sanitizeGemmaMarkdown', () {
    test('removes inline dollar delimiters around numeric values', () {
      final text =
          'The primary findings are: Pitch Variability: \$141.7237\$, Voiced fraction: \$0.9249\$.';
      final sanitized = sanitizeGemmaMarkdown(text);
      expect(
        sanitized,
        'The primary findings are: Pitch Variability: 141.7237, Voiced fraction: 0.9249.',
      );
    });

    test('preserves normal prose without numeric math delimiters', () {
      final text = 'This result should be discussed with a doctor.';
      expect(sanitizeGemmaMarkdown(text), text);
    });
  });
}
