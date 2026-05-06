import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';

import '../l10n/app_strings.dart';
import '../models/doctor_discussion_guide.dart';
import '../services/gemma_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../utils/markdown_utils.dart';

/// Bottom sheet for rehearsing a clinician conversation with Gemma as coach.
///
/// Opens as a [DraggableScrollableSheet]. Gemma speaks first by simulating
/// questions a clinician might ask. Supports multi-turn conversation with
/// suggested response chips and an end-practice summary.
///
/// Props:
///   [guide]    - the generated [DoctorDiscussionGuide] to practice against.
///   [language] - BCP-47 language tag used for all l10n and Gemma prompts.
class DoctorPracticeSheet extends StatefulWidget {
  const DoctorPracticeSheet({
    super.key,
    required this.guide,
    required this.language,
  });

  final DoctorDiscussionGuide guide;
  final String language;

  @override
  State<DoctorPracticeSheet> createState() => _DoctorPracticeSheetState();
}

class _DoctorPracticeSheetState extends State<DoctorPracticeSheet> {
  // ── Message list ───────────────────────────────────────────────────────────
  // Roles: 'coach' (Gemma) or 'patient' (the user).
  final List<({String role, String text})> _messages = [];
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  bool _isGenerating = false;
  String _streamBuffer = '';
  bool _hasEnded = false;
  String? _summary;
  List<String> _suggestedResponses = [];

  // ── i18n helper ───────────────────────────────────────────────────────────
  String _s(String key) => AppStrings.get(key, widget.language);

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    _startConversation();
  }

  @override
  void dispose() {
    GemmaService.resetRehearsal();
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // ── Conversation methods ──────────────────────────────────────────────────

  /// Streams the coach opening question. Called once from [initState].
  Future<void> _startConversation() async {
    setState(() {
      _isGenerating = true;
      _streamBuffer = '';
    });

    try {
      await for (final token in GemmaService.startRehearsalStream(
        guide: widget.guide,
        language: widget.language,
      )) {
        if (!mounted) return;
        setState(() => _streamBuffer += token);
        _scrollToBottom();
      }
      if (!mounted) return;
      setState(() {
        _messages.add((role: 'coach', text: _streamBuffer.trim()));
        _streamBuffer = '';
        _isGenerating = false;
        _suggestedResponses = GemmaService.suggestedRehearsalResponses(
          widget.language,
        );
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _messages.add((role: 'coach', text: 'Error: $e'));
        _streamBuffer = '';
        _isGenerating = false;
      });
    }
    _scrollToBottom();
  }

  /// Sends a user message and streams the coach reply.
  Future<void> _send(String text) async {
    if (text.trim().isEmpty || _isGenerating || _hasEnded) return;
    setState(() {
      _messages.add((role: 'patient', text: text.trim()));
      _textController.clear();
      _isGenerating = true;
      _streamBuffer = '';
      _suggestedResponses = [];
    });
    _scrollToBottom();

    try {
      await for (final token in GemmaService.continueRehearsalStream(
        text.trim(),
      )) {
        if (!mounted) return;
        setState(() => _streamBuffer += token);
        _scrollToBottom();
      }
      if (!mounted) return;
      setState(() {
        _messages.add((role: 'coach', text: _streamBuffer.trim()));
        _streamBuffer = '';
        _isGenerating = false;
        _suggestedResponses = GemmaService.suggestedRehearsalResponses(
          widget.language,
        );
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _messages.add((role: 'coach', text: 'Error: $e'));
        _streamBuffer = '';
        _isGenerating = false;
      });
    }
    _scrollToBottom();
  }

  /// Ends the session and streams the practice summary.
  Future<void> _endPractice() async {
    if (_isGenerating || _hasEnded) return;
    setState(() {
      _isGenerating = true;
      _streamBuffer = '';
      _suggestedResponses = [];
    });
    _scrollToBottom();

    try {
      final buffer = StringBuffer();
      await for (final token in GemmaService.endRehearsalStream()) {
        if (!mounted) return;
        buffer.write(token);
        setState(() => _streamBuffer = buffer.toString());
        _scrollToBottom();
      }
      if (!mounted) return;
      setState(() {
        _summary = buffer.toString().trim();
        _streamBuffer = '';
        _isGenerating = false;
        _hasEnded = true;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _summary = 'Error: $e';
        _streamBuffer = '';
        _isGenerating = false;
        _hasEnded = true;
      });
    }
    _scrollToBottom();
  }

  /// Routes a chip tap: the end-practice chip calls [_endPractice],
  /// all others call [_send].
  void _onChipTap(String text) {
    final endPrompt = AppStrings.get('practice_end_prompt', widget.language);
    if (text == endPrompt) {
      _endPractice();
    } else {
      _send(text);
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  // ── List helpers ──────────────────────────────────────────────────────────

  /// Total item count: completed messages + optional streaming row + optional
  /// summary card. The two extras are mutually exclusive (_isGenerating and
  /// _hasEnded cannot both be true simultaneously).
  int get _itemCount {
    int count = _messages.length;
    if (_isGenerating) count += 1;
    if (_hasEnded && _summary != null) count += 1;
    return count;
  }

  Widget _buildItem(int index) {
    if (index < _messages.length) {
      final msg = _messages[index];
      if (msg.role == 'coach') return _CoachBubble(text: msg.text);
      return _PatientBubble(text: msg.text);
    }
    // Extra slot: streaming bubble while generating, summary card when ended.
    if (_isGenerating) {
      return _PracticeStreamingBubble(
        buffer: _streamBuffer,
        thinkingLabel: _s('practice_generating'),
      );
    }
    if (_hasEnded && _summary != null) {
      return _SummaryCard(
        title: _s('practice_summary_title'),
        summary: _summary!,
      );
    }
    return const SizedBox.shrink();
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      initialChildSize: 0.92,
      minChildSize: 0.60,
      maxChildSize: 0.96,
      expand: false,
      builder: (context, sheetScrollController) {
        return Container(
          decoration: const BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Padding(
            padding: EdgeInsets.only(
              bottom: MediaQuery.viewInsetsOf(context).bottom,
            ),
            child: Column(
              children: [
                // ── Drag handle ───────────────────────────────────────────
                const SizedBox(height: 12),
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
                const SizedBox(height: 12),

                // ── Header row ────────────────────────────────────────────
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: Row(
                    children: [
                      const Icon(
                        Icons.assignment_outlined,
                        size: 20,
                        color: AppColors.primary,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _s('practice_title'),
                          style: AppTextStyles.headingSmall,
                        ),
                      ),
                      if (!_hasEnded && !_isGenerating)
                        TextButton(
                          onPressed: _messages.length >= 2
                              ? _endPractice
                              : null,
                          style: TextButton.styleFrom(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 4,
                            ),
                            minimumSize: Size.zero,
                            tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                          ),
                          child: Text(
                            _s('end_practice'),
                            style: AppTextStyles.caption.copyWith(
                              color: _messages.length >= 2
                                  ? AppColors.primary
                                  : AppColors.textTertiary,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                      IconButton(
                        icon: const Icon(Icons.close_rounded, size: 22),
                        color: AppColors.textSecondary,
                        onPressed: () => Navigator.of(context).pop(),
                        padding: EdgeInsets.zero,
                        constraints: const BoxConstraints(),
                      ),
                    ],
                  ),
                ),

                const Divider(height: 1, color: AppColors.border),

                // ── Messages list ─────────────────────────────────────────
                Expanded(
                  child: ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
                    itemCount: _itemCount,
                    itemBuilder: (context, index) => _buildItem(index),
                  ),
                ),

                // ── Suggested response chips (horizontal scroll) ──────────
                if (_suggestedResponses.isNotEmpty && !_isGenerating)
                  _SuggestedResponseChips(
                    responses: _suggestedResponses,
                    onTap: _onChipTap,
                  ),

                // ── Input row (hidden after practice ends) ────────────────
                if (!_hasEnded) ...[
                  const Divider(height: 1, color: AppColors.border),
                  _PracticeInputBar(
                    controller: _textController,
                    isGenerating: _isGenerating,
                    placeholder: _s('chat_placeholder'),
                    onSend: _send,
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
}

// ── Coach bubble (left-aligned, Gemma's messages) ─────────────────────────────

class _CoachBubble extends StatelessWidget {
  const _CoachBubble({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    final sanitized = sanitizeGemmaMarkdown(text);
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 28,
            height: 28,
            margin: const EdgeInsets.only(right: 8, top: 2),
            decoration: BoxDecoration(
              color: AppColors.primaryContainer,
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.assignment_outlined,
              size: 14,
              color: AppColors.primary,
            ),
          ),
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: AppColors.surfaceElevated,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(16),
                  topRight: Radius.circular(16),
                  bottomRight: Radius.circular(16),
                  bottomLeft: Radius.circular(4),
                ),
                border: Border.all(color: AppColors.border),
              ),
              child: MarkdownBody(
                data: sanitized,
                styleSheet: MarkdownStyleSheet(
                  p: AppTextStyles.bodyMedium.copyWith(
                    color: AppColors.textPrimary,
                    height: 1.5,
                  ),
                  h2: AppTextStyles.headingSmall.copyWith(
                    color: AppColors.textPrimary,
                  ),
                  h3: AppTextStyles.bodyMedium.copyWith(
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary,
                  ),
                  strong: AppTextStyles.bodyMedium.copyWith(
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary,
                  ),
                  listBullet: AppTextStyles.bodyMedium.copyWith(
                    color: AppColors.textPrimary,
                    height: 1.5,
                  ),
                  blockSpacing: 8,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Patient bubble (right-aligned, user's messages) ───────────────────────────

class _PatientBubble extends StatelessWidget {
  const _PatientBubble({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: AppColors.primaryContainer,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(16),
                  topRight: Radius.circular(16),
                  bottomLeft: Radius.circular(16),
                  bottomRight: Radius.circular(4),
                ),
              ),
              child: Text(
                text,
                style: AppTextStyles.bodyMedium.copyWith(
                  color: AppColors.primaryDark,
                  height: 1.5,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Streaming bubble (live token accumulation, coach side) ────────────────────

class _PracticeStreamingBubble extends StatelessWidget {
  const _PracticeStreamingBubble({
    required this.buffer,
    required this.thinkingLabel,
  });

  final String buffer;
  final String thinkingLabel;

  @override
  Widget build(BuildContext context) {
    final sanitized = sanitizeGemmaMarkdown(buffer);
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 28,
            height: 28,
            margin: const EdgeInsets.only(right: 8, top: 2),
            decoration: const BoxDecoration(
              color: AppColors.primaryContainer,
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.assignment_outlined,
              size: 14,
              color: AppColors.primary,
            ),
          ),
          Flexible(
            child: AnimatedSize(
              duration: const Duration(milliseconds: 100),
              alignment: Alignment.topLeft,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 10,
                ),
                decoration: BoxDecoration(
                  color: AppColors.surfaceElevated,
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(16),
                    topRight: Radius.circular(16),
                    bottomRight: Radius.circular(16),
                    bottomLeft: Radius.circular(4),
                  ),
                  border: Border.all(color: AppColors.border),
                ),
                child: buffer.isEmpty
                    ? Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(
                              strokeWidth: 1.5,
                              color: AppColors.primary.withValues(alpha: 0.63),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            thinkingLabel,
                            style: AppTextStyles.caption.copyWith(
                              color: AppColors.textTertiary,
                            ),
                          ),
                        ],
                      )
                    : Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          MarkdownBody(
                            data: sanitized,
                            styleSheet: MarkdownStyleSheet(
                              p: AppTextStyles.bodyMedium.copyWith(height: 1.5),
                              h2: AppTextStyles.headingSmall,
                              h3: AppTextStyles.bodyMedium.copyWith(
                                fontWeight: FontWeight.w600,
                              ),
                              strong: AppTextStyles.bodyMedium.copyWith(
                                fontWeight: FontWeight.w600,
                              ),
                              listBullet: AppTextStyles.bodyMedium.copyWith(
                                height: 1.5,
                              ),
                              blockSpacing: 8,
                            ),
                          ),
                          // Blinking cursor bar
                          Container(
                            width: 2,
                            height: 16,
                            margin: const EdgeInsets.only(top: 4),
                            color: AppColors.primary,
                          ),
                        ],
                      ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Summary card (shown after practice ends) ──────────────────────────────────

class _SummaryCard extends StatelessWidget {
  const _SummaryCard({required this.title, required this.summary});

  final String title;
  final String summary;

  @override
  Widget build(BuildContext context) {
    final sanitized = sanitizeGemmaMarkdown(summary);
    return Padding(
      padding: const EdgeInsets.only(top: 4, bottom: 12),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
        decoration: BoxDecoration(
          color: AppColors.primary.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.primary.withValues(alpha: 0.20)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(
                  Icons.check_circle_outline_rounded,
                  size: 16,
                  color: AppColors.primary,
                ),
                const SizedBox(width: 6),
                Text(
                  title,
                  style: AppTextStyles.captionStrong.copyWith(
                    color: AppColors.primary,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            MarkdownBody(
              data: sanitized,
              styleSheet: MarkdownStyleSheet(
                p: AppTextStyles.bodyMedium.copyWith(
                  color: AppColors.textPrimary,
                  height: 1.5,
                ),
                h3: AppTextStyles.bodyMedium.copyWith(
                  fontWeight: FontWeight.w600,
                  color: AppColors.textPrimary,
                ),
                strong: AppTextStyles.bodyMedium.copyWith(
                  fontWeight: FontWeight.w600,
                  color: AppColors.textPrimary,
                ),
                listBullet: AppTextStyles.bodyMedium.copyWith(
                  color: AppColors.textPrimary,
                  height: 1.5,
                ),
                blockSpacing: 8,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Suggested response chips (horizontal scroll row) ──────────────────────────

class _SuggestedResponseChips extends StatelessWidget {
  const _SuggestedResponseChips({required this.responses, required this.onTap});

  final List<String> responses;
  final void Function(String) onTap;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
      child: Row(
        children: responses.map((text) {
          return Padding(
            padding: const EdgeInsets.only(right: 8),
            child: ActionChip(
              label: Text(text, maxLines: 2, overflow: TextOverflow.ellipsis),
              onPressed: () => onTap(text),
              labelStyle: AppTextStyles.caption.copyWith(
                color: AppColors.primary,
                fontWeight: FontWeight.w500,
              ),
              backgroundColor: AppColors.surface,
              side: const BorderSide(color: AppColors.primary),
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
            ),
          );
        }).toList(),
      ),
    );
  }
}

// ── Text input bar ────────────────────────────────────────────────────────────

class _PracticeInputBar extends StatelessWidget {
  const _PracticeInputBar({
    required this.controller,
    required this.isGenerating,
    required this.placeholder,
    required this.onSend,
  });

  final TextEditingController controller;
  final bool isGenerating;
  final String placeholder;
  final Future<void> Function(String) onSend;

  @override
  Widget build(BuildContext context) {
    return Container(
      color: AppColors.surface,
      padding: const EdgeInsets.fromLTRB(16, 10, 12, 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Expanded(
            child: Container(
              constraints: const BoxConstraints(maxHeight: 120),
              decoration: BoxDecoration(
                color: AppColors.surfaceElevated,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppColors.border),
              ),
              child: TextField(
                controller: controller,
                enabled: !isGenerating,
                maxLines: null,
                keyboardType: TextInputType.multiline,
                textInputAction: TextInputAction.newline,
                style: AppTextStyles.bodyMedium,
                decoration: InputDecoration(
                  hintText: placeholder,
                  hintStyle: AppTextStyles.bodyMedium.copyWith(
                    color: AppColors.textTertiary,
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 10,
                  ),
                  border: InputBorder.none,
                  enabledBorder: InputBorder.none,
                  focusedBorder: InputBorder.none,
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),
          _PracticeSendButton(
            isGenerating: isGenerating,
            onTap: () => onSend(controller.text),
          ),
        ],
      ),
    );
  }
}

class _PracticeSendButton extends StatelessWidget {
  const _PracticeSendButton({required this.isGenerating, required this.onTap});

  final bool isGenerating;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: isGenerating ? null : onTap,
      child: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: isGenerating ? AppColors.border : AppColors.primary,
          shape: BoxShape.circle,
        ),
        child: isGenerating
            ? const Padding(
                padding: EdgeInsets.all(11),
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: AppColors.textTertiary,
                ),
              )
            : const Icon(
                Icons.arrow_upward_rounded,
                color: AppColors.onPrimary,
                size: 20,
              ),
      ),
    );
  }
}
