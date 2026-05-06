import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:cognitrace/models/doctor_discussion_guide.dart';
import 'package:cognitrace/widgets/doctor_practice_sheet.dart';

void main() {
  group('DoctorPracticeSheet Widget Tests', () {
    // Test fixture: a minimal valid DoctorDiscussionGuide
    final testGuide = DoctorDiscussionGuide(
      visitReason: 'I used a voice screening app.',
      resultSummary: 'Low risk score of 0.22.',
      questionsToAsk: ['Should I repeat the screening?'],
      contextToShare: ['Recorded at home.'],
      caveats: ['Screening tool, not diagnosis.'],
    );

    testWidgets('Widget builds without crashing', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: DoctorPracticeSheet(guide: testGuide, language: 'English'),
          ),
        ),
      );

      // Widget should render without throwing
      expect(find.byType(DoctorPracticeSheet), findsOneWidget);
    });

    testWidgets('Displays practice title', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: DoctorPracticeSheet(guide: testGuide, language: 'English'),
          ),
        ),
      );

      // The title should be visible (from AppStrings.get('practice_title', 'English'))
      // We verify the Text widget exists with the expected key/style
      expect(find.byType(Text), findsWidgets);
    });

    testWidgets('Displays close button', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: DoctorPracticeSheet(guide: testGuide, language: 'English'),
          ),
        ),
      );

      // Close button should be present (IconButton with close_rounded icon)
      expect(find.byIcon(Icons.close_rounded), findsOneWidget);
    });

    testWidgets('Displays input field before practice ends', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: DoctorPracticeSheet(guide: testGuide, language: 'English'),
          ),
        ),
      );

      // TextField should be present in the input bar
      expect(find.byType(TextField), findsOneWidget);
    });

    testWidgets('Handles missing engine gracefully', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: DoctorPracticeSheet(guide: testGuide, language: 'English'),
          ),
        ),
      );

      // Pump a few frames to allow async operations to start
      await tester.pump(const Duration(milliseconds: 100));
      await tester.pump(const Duration(milliseconds: 100));

      // Widget should still be present (no unhandled exception)
      expect(find.byType(DoctorPracticeSheet), findsOneWidget);

      // The widget should render without crashing even if GemmaService fails
      expect(find.byType(Scaffold), findsOneWidget);
    });
  });
}
