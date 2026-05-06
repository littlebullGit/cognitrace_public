/// Strip LaTeX wrappers and dollar-sign math notation from Gemma output.
String sanitizeGemmaMarkdown(String text) {
  // Strip LaTeX \text{} wrappers first: $\text{f0_std}$ -> f0_std
  var result = text.replaceAllMapped(
    RegExp(r'\$\\text\{([^}]*)\}\$'),
    (match) => match.group(1) ?? match.group(0)!,
  );
  // Strip remaining $...$ (numbers, percentages, other LaTeX)
  result = result.replaceAllMapped(
    RegExp(r'\$([^$]+)\$'),
    (match) => match.group(1) ?? match.group(0)!,
  );
  return result;
}
