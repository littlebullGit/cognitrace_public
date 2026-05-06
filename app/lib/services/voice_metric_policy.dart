class VoiceMetricPolicy {
  static const _referenceThresholds =
      <String, ({double healthyRaw, bool higherIsBad, String label})>{
        // Classic sustained-vowel reference flags carried over as soft guidance.
        // Internal comparison stays in raw ratio form; labels remain percentages.
        'jitter_local': (
          healthyRaw: 0.0104,
          higherIsBad: true,
          label: 'classic sustained-vowel ref <1.04%',
        ),
        'jitter_rap': (
          healthyRaw: 0.0068,
          higherIsBad: true,
          label: 'classic sustained-vowel ref <0.68%',
        ),
      };

  static bool hasReferenceThreshold(String key) =>
      _referenceThresholds.containsKey(key);

  static bool isReferenceHigh(String key, double value) {
    final threshold = _referenceThresholds[key];
    if (threshold == null) return false;
    return threshold.higherIsBad
        ? value > threshold.healthyRaw
        : value < threshold.healthyRaw;
  }

  static String annotation(String key, double value) {
    final threshold = _referenceThresholds[key];
    if (threshold == null) return '';
    final status = isReferenceHigh(key, value) ? 'ABOVE REF' : 'WITHIN REF';
    return ' [$status vs ${threshold.label}]';
  }

  static String formatValue(String key, double value) {
    if (key.startsWith('jitter_') || key.startsWith('shimmer_')) {
      return '${(value * 100).toStringAsFixed(2)}%';
    }
    if (key == 'hnr') return '${value.toStringAsFixed(2)} dB';
    if (value.abs() >= 100) return value.toStringAsFixed(1);
    if (value.abs() >= 1) return value.toStringAsFixed(3);
    return value.toStringAsFixed(4);
  }
}
