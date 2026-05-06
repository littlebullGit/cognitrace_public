import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';

import '../l10n/app_strings.dart';
import '../services/gemma_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../utils/markdown_utils.dart';

/// Full-featured Gemma follow-up chat bottom sheet.
///
/// Opens as a [DraggableScrollableSheet] from the results screen.
/// Streams tokens from [GemmaService.askFollowUpStream] and auto-scrolls
/// to the bottom as each token arrives.
class GemmaChatSheet extends StatefulWidget {
  const GemmaChatSheet({super.key, required this.language});

  final String language;

  @override
  State<GemmaChatSheet> createState() => _GemmaChatSheetState();
}

class _GemmaChatSheetState extends State<GemmaChatSheet> {
  // ── Message model ─────────────────────────────────────────────────────────
  final List<({String role, String text})> _messages = [];
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  bool _isGenerating = false;
  String _streamBuffer = '';

  // ── i18n helper ───────────────────────────────────────────────────────────
  String _s(String key) => AppStrings.get(key, widget.language);

  // ── Suggested questions ───────────────────────────────────────────────────
  List<String> get _suggestedQuestions {
    final dynamic = GemmaService.suggestedQuestions(widget.language);
    if (dynamic.isNotEmpty) return dynamic;
    return [
      _s('suggested_detailed'),
      _s('suggested_jitter'),
      _s('suggested_doctor'),
      _s('suggested_improve'),
      _s('suggested_compare'),
    ];
  }

  // ── Whether to show suggested chips ──────────────────────────────────────
  bool get _showSuggestions {
    if (_isGenerating) return false;
    if (_messages.isEmpty) return true;
    // Show after a Gemma response completes.
    return _messages.isNotEmpty && _messages.last.role == 'gemma';
  }

  // ── Send message ──────────────────────────────────────────────────────────
  Future<void> _send(String text) async {
    if (text.trim().isEmpty || _isGenerating) return;
    setState(() {
      _messages.add((role: 'user', text: text.trim()));
      _textController.clear();
      _isGenerating = true;
      _streamBuffer = '';
    });
    _scrollToBottom();

    try {
      await for (final token in GemmaService.askFollowUpStream(text.trim())) {
        if (!mounted) return;
        setState(() => _streamBuffer += token);
        _scrollToBottom();
      }
      if (!mounted) return;
      setState(() {
        _messages.add((role: 'gemma', text: _streamBuffer.trim()));
        _streamBuffer = '';
        _isGenerating = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _messages.add((role: 'gemma', text: 'Error: $e'));
        _streamBuffer = '';
        _isGenerating = false;
      });
    }
    _scrollToBottom();
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

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // ── Build ─────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      initialChildSize: 0.70,
      minChildSize: 0.40,
      maxChildSize: 0.95,
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

                // ── Header ────────────────────────────────────────────────
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: Row(
                    children: [
                      const Icon(
                        Icons.auto_awesome_rounded,
                        size: 18,
                        color: AppColors.primary,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _s('ask_gemma'),
                          style: AppTextStyles.headingSmall,
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

                // ── Disclaimer ────────────────────────────────────────────
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 6, 20, 8),
                  child: Text(
                    _s('chat_disclaimer'),
                    style: AppTextStyles.caption.copyWith(
                      color: AppColors.textTertiary,
                      fontSize: 12,
                    ),
                  ),
                ),

                const Divider(height: 1, color: AppColors.border),

                // ── Messages list ─────────────────────────────────────────
                Expanded(
                  child: ListView(
                    controller: _scrollController,
                    padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
                    children: [
                      // Render completed messages
                      ..._messages.map(
                        (msg) => _MessageBubble(role: msg.role, text: msg.text),
                      ),

                      // Streaming bubble (while generating)
                      if (_isGenerating)
                        _StreamingBubble(
                          buffer: _streamBuffer,
                          thinkingLabel: _s('chat_thinking'),
                        ),

                      // Suggested chips
                      if (_showSuggestions)
                        _SuggestedChips(
                          questions: _suggestedQuestions,
                          onTap: _send,
                        ),

                      const SizedBox(height: 8),
                    ],
                  ),
                ),

                // ── Input row ────────────────────────────────────────────
                const Divider(height: 1, color: AppColors.border),
                _ChatInputBar(
                  controller: _textController,
                  isGenerating: _isGenerating,
                  placeholder: _s('chat_placeholder'),
                  onSend: _send,
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

// ── Message bubble ────────────────────────────────────────────────────────────

class _MessageBubble extends StatelessWidget {
  const _MessageBubble({required this.role, required this.text});

  final String role;
  final String text;

  bool get _isUser => role == 'user';

  @override
  Widget build(BuildContext context) {
    final sanitizedText = sanitizeGemmaMarkdown(text);
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: _isUser
            ? MainAxisAlignment.end
            : MainAxisAlignment.start,
        children: [
          if (!_isUser) ...[
            Container(
              width: 28,
              height: 28,
              margin: const EdgeInsets.only(right: 8, top: 2),
              decoration: BoxDecoration(
                color: AppColors.primaryContainer,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.auto_awesome,
                size: 14,
                color: AppColors.primary,
              ),
            ),
          ],
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: _isUser
                    ? AppColors.primaryContainer
                    : AppColors.surfaceElevated,
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(16),
                  topRight: const Radius.circular(16),
                  bottomLeft: Radius.circular(_isUser ? 16 : 4),
                  bottomRight: Radius.circular(_isUser ? 4 : 16),
                ),
                border: _isUser ? null : Border.all(color: AppColors.border),
              ),
              child: _isUser
                  ? Text(
                      text,
                      style: AppTextStyles.bodyMedium.copyWith(
                        color: AppColors.primaryDark,
                        height: 1.5,
                      ),
                    )
                  : MarkdownBody(
                      data: sanitizedText,
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

// ── Streaming bubble (live token accumulation) ────────────────────────────────

class _StreamingBubble extends StatelessWidget {
  const _StreamingBubble({required this.buffer, required this.thinkingLabel});

  final String buffer;
  final String thinkingLabel;

  @override
  Widget build(BuildContext context) {
    final sanitizedBuffer = sanitizeGemmaMarkdown(buffer);
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
              Icons.auto_awesome,
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
                              color: AppColors.primary.withAlpha(160),
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
                            data: sanitizedBuffer,
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

// ── Suggested question chips ──────────────────────────────────────────────────

class _SuggestedChips extends StatelessWidget {
  const _SuggestedChips({required this.questions, required this.onTap});

  final List<String> questions;
  final void Function(String) onTap;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 4, bottom: 8),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: questions.map((q) {
          return ActionChip(
            label: Text(q),
            onPressed: () => onTap(q),
            labelStyle: AppTextStyles.caption.copyWith(
              color: AppColors.primary,
              fontWeight: FontWeight.w500,
            ),
            backgroundColor: AppColors.surface,
            side: const BorderSide(color: AppColors.primary),
            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
            materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
          );
        }).toList(),
      ),
    );
  }
}

// ── Text input bar ────────────────────────────────────────────────────────────

class _ChatInputBar extends StatelessWidget {
  const _ChatInputBar({
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
          _SendButton(
            isGenerating: isGenerating,
            onTap: () => onSend(controller.text),
          ),
        ],
      ),
    );
  }
}

class _SendButton extends StatelessWidget {
  const _SendButton({required this.isGenerating, required this.onTap});

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
