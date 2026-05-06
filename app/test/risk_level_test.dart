import 'package:flutter_test/flutter_test.dart';

// Mirrors the risk level logic in GemmaService.analyze()
String riskLevel(double score) => score >= 0.7
    ? 'elevated'
    : score >= 0.4
    ? 'moderate'
    : 'low';

void main() {
  group('Risk level classification', () {
    test('score 0.0 is low', () => expect(riskLevel(0.0), 'low'));
    test('score 0.25 is low', () => expect(riskLevel(0.25), 'low'));
    test('score 0.39 is low', () => expect(riskLevel(0.39), 'low'));
    test('score 0.4 is moderate', () => expect(riskLevel(0.4), 'moderate'));
    test('score 0.55 is moderate', () => expect(riskLevel(0.55), 'moderate'));
    test('score 0.69 is moderate', () => expect(riskLevel(0.69), 'moderate'));
    test('score 0.7 is elevated', () => expect(riskLevel(0.7), 'elevated'));
    test('score 0.85 is elevated', () => expect(riskLevel(0.85), 'elevated'));
    test('score 1.0 is elevated', () => expect(riskLevel(1.0), 'elevated'));
  });

  group('Risk level boundary conditions', () {
    test('score just below 0.4 is low', () => expect(riskLevel(0.3999), 'low'));
    test(
      'score just above 0.4 is moderate',
      () => expect(riskLevel(0.4001), 'moderate'),
    );
    test(
      'score just below 0.7 is moderate',
      () => expect(riskLevel(0.6999), 'moderate'),
    );
    test(
      'score just above 0.7 is elevated',
      () => expect(riskLevel(0.7001), 'elevated'),
    );
  });
}
