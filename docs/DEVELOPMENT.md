# CogniTrace Development & Packaging Guide

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Flutter | 3.41+ (Dart 3.11+) | `flutter --version` to check |
| Xcode | 16.0+ | Required for iOS build toolchain |
| CocoaPods | 1.15+ | `sudo gem install cocoapods` if missing |
| Physical iPhone | iOS 16.4+ | **No simulator support**. llamadart FFI requires real Metal GPU |
| Apple Developer account | Free tier works | Team ID configured in Xcode signing |

**No API keys, tokens, or paid services required.** The Gemma 4 GGUF model is public Apache 2.0 on HuggingFace.

---

## Repository Layout

```
cognitrace/
├── app/                          # Flutter project root
│   ├── lib/
│   │   ├── main.dart             # App entry point, eager Gemma download
│   │   ├── screens/              # 7 screens (splash → onboarding → home → record → analysis → results → settings)
│   │   ├── services/             # Audio, inference, Gemma, history, i18n
│   │   ├── l10n/                 # Static i18n (5 languages, ~155 keys)
│   │   ├── models/               # Data models (CheckRecord)
│   │   ├── navigation/           # Route definitions
│   │   └── theme/                # Colors, typography, Material theme
│   ├── ios/
│   │   ├── Runner/
│   │   │   ├── AppDelegate.swift
│   │   │   ├── AudioBridge.swift       # Platform channel: recording + playback
│   │   │   ├── FeatureExtractor.swift  # 56-biomarker extraction (vDSP/Accelerate)
│   │   │   └── FFTProcessor.swift      # FFT, spectral features, MFCC
│   │   └── Podfile                     # Critical linker flags for llamadart
│   └── assets/
│       ├── models/               # ONNX ensemble models + scaler params
│       ├── reference_audio/      # Bundled test clips (PD + control)
│       └── icon/                 # App icon (brain + waveform)
├── docs/                         # Public development, privacy, and product specs
└── README.md                     # Project overview
```

Internal judge prep, demo scripts, App Store Connect notes, and planning
roadmaps are kept outside this public repository.

---

## First-Time Setup

```bash
# 1. Clone
git clone https://github.com/littlebullGit/cognitrace_public.git
cd cognitrace_public/app

# 2. Install Flutter dependencies
flutter pub get

# 3. Install iOS native dependencies
cd ios && pod install && cd ..
```

### Xcode Signing

Open `ios/Runner.xcworkspace` in Xcode (not `.xcodeproj`):

1. Select the **Runner** target
2. Go to **Signing & Capabilities**
3. Set **Team** to your Apple Developer account
4. Bundle ID is `com.cognitrace.cognitrace`. Change it if you get signing conflicts
5. Xcode will auto-provision a development certificate

Use your own Apple Developer team for local signing. Do not commit personal
signing IDs or release-account metadata.

---

## Running the App

### Debug Mode (Xcode Play Button)

Best for development: hot reload, debug logs, breakpoints.

```bash
# From terminal:
flutter run

# Or: open ios/Runner.xcworkspace in Xcode → select device → Play (Cmd+R)
```

- Hot reload: `r` in terminal, or Cmd+S in IDE
- Hot restart: `R` in terminal
- Debug overhead: ~400 MB extra memory vs release
- Gemma inference may be slower in debug

### Release Mode (Demo / Testing)

Recommended for real device testing and demos. Significantly faster, less memory.

```bash
flutter run --release
```

### Profile Mode (Performance Debugging)

```bash
flutter run --profile
```

Useful for debugging Gemma inference speed or feature extraction timing.

---

## Gemma 4 Model Download

The GGUF model is **not bundled** in the app. It downloads automatically on first launch, but it is only required for AI interpretation. Recording, feature extraction, and ONNX scoring can run while the model is still downloading.

| Detail | Value |
|--------|-------|
| Model | Gemma 4 E2B (Q4_K_M quantization) |
| Size | ~2.7 GB |
| Source | `huggingface.co/littlebull9/cognitrace-gemma4-medical-GGUF` |
| Storage | `Documents/cognitrace-gemma4-medical-v3-Q4_K_M.gguf` |
| Download time | ~4 min on WiFi |
| Auth | None required (Apache 2.0) |

**Flow:**
1. `main.dart` calls `GemmaDownloadManager.instance.ensureStarted()` eagerly
2. Home screen shows download progress card while downloading
3. Users can record and score checks before the download finishes
4. Results screen auto-generates narrative once the model is ready

If iOS suspends the app while the screen is locked, the next foreground retry
keeps the `.tmp` file and resumes via HTTP range requests instead of starting
from zero when the server supports ranges.

**Re-download:** Settings screen -> "Re-download model" or delete `Documents/cognitrace-gemma4-medical-v3-Q4_K_M.gguf` from the device.

**Offline development:** Once downloaded, the model persists across app restarts. No network needed for subsequent runs.

---

## Key Build Configuration

### Podfile Linker Flags (CRITICAL)

The Podfile at `ios/Podfile` contains linker flags that **must not be removed**:

```ruby
ld_flags << '-all_load'           # Load all symbols from static libs
ld_flags << '-Wl,-export_dynamic' # Export symbols for FFI
config.build_settings['STRIP_STYLE'] = 'non-global'
```

Without these, llamadart's llama.cpp symbols get stripped during linking and the engine fails silently at runtime.

### iOS Deployment Target

Set to **16.4** in both the Xcode project and the Podfile post_install hook. llamadart FFI requires 16.4+.

### Info.plist Permissions

| Key | Purpose |
|-----|---------|
| `NSMicrophoneUsageDescription` | Voice recording for analysis |
| `NSLocalNetworkUsageDescription` | Required by llamadart for model inference |
| `UIFileSharingEnabled` | Disabled for release so the app container is not exposed through iTunes or Files |
| `ITSAppUsesNonExemptEncryption` | Set to `false` so App Store Connect knows the app does not use non-exempt encryption |

---

## Build Modes & Memory

| Mode | Memory | Gemma Inference | Use Case |
|------|--------|-----------------|----------|
| Debug | ~1.4 GB | ~25-30s | Development, hot reload |
| Release | ~1.0 GB | ~10-15s | Demo, real testing |
| Profile | ~1.1 GB | ~12-18s | Performance analysis |

**iPhone 13 (4 GB RAM):** Tight. Use release mode. May need `gpuLayers` tuning in `gemma_service.dart` if OOM occurs.

**iPhone 14 (6 GB RAM):** Comfortable in all modes.

### Gemma Engine Parameters

In `lib/services/gemma_service.dart`, line 121:

```dart
modelParams: ModelParams(
  contextSize: 2048,     // Supports initial narrative + multi-turn chat
  batchSize: 512,        // Reduce compute buffer
  microBatchSize: 256,   // Critical: matches Gemma head_size=256
  gpuLayers: 999,        // All 35 layers on Metal GPU
  preferredBackend: GpuBackend.metal,
),
```

Tuning `contextSize` down to 1024 or `gpuLayers` down to 28-30 can help on memory-constrained devices.

---

## Release Build

For local release validation, build an IPA from `app/`:

```bash
flutter clean
flutter pub get
cd ios && pod install && cd ..
flutter build ipa --release
```

The Xcode build also runs `Runner/Scripts/fix_native_asset_plists.sh` after
Flutter embeds native assets. This keeps `llamadart.framework`'s
`MinimumOSVersion` aligned with the Runner deployment target and emits dSYMs for
native asset frameworks.

This produces `build/ios/archive/Runner.xcarchive`.

TestFlight, App Store Connect metadata, private signing IDs, and release notes
are intentionally kept outside this public repository.

---

## Common Issues

### "No provisioning profile" / signing errors

Open `ios/Runner.xcworkspace` in Xcode, go to Signing & Capabilities, and set your Team. Free accounts work but expire after 7 days.

### Pod install fails

```bash
cd ios
pod deintegrate
pod cache clean --all
pod install
```

### llamadart symbols not found at runtime

The Podfile linker flags were likely removed. Verify `ios/Podfile` contains `-all_load` and `-Wl,-export_dynamic` in the post_install block.

### Gemma download stalls or fails

- Check WiFi connectivity (about 2.7 GB download)
- Partial downloads are preserved automatically; reopen the app or tap "Retry download" to resume
- Check device storage (need several GB free for the model plus temp file during download)

### "Metal library not found" or GPU errors

- Ensure running on **physical device**, not simulator
- Metal is not available in iOS simulator
- Try `flutter clean && flutter run --release`

### Audio recording returns silence

- Grant microphone permission when prompted
- Check Settings → CogniTrace → Microphone is enabled
- Ensure no other app is using the microphone

### OOM crash during Gemma inference (iPhone 13)

Reduce memory in `lib/services/gemma_service.dart`:

```dart
contextSize: 1024,    // Down from 2048
gpuLayers: 28,       // Down from 999 (offload some to CPU)
```

### Xcode container UUID changes break audio playback

This is handled automatically. Audio paths are stored as relative (`saved_audio_runs/123.wav`) and resolved dynamically via `getApplicationDocumentsDirectory()`.

---

## Static Analysis

```bash
cd app

# Dart analysis (should show "No issues found!")
flutter analyze

# Format check
dart format --set-exit-if-changed lib/
```

---

## Project Conventions

- **Async everywhere:** All services use `async`/`await`. No sync I/O.
- **i18n:** Static map in `l10n/app_strings.dart`. Gemma handles dynamic content; AppStrings handles static UI.
- **Platform channels:** Swift code in `ios/Runner/` communicates via `MethodChannel('com.cognitrace.audio')`.
- **State management:** `ChangeNotifier` (GemmaDownloadManager) + `StatefulWidget` state. No Riverpod/Bloc.
- **File paths:** Always relative for persistence (survives iOS container UUID changes on Xcode rebuild).
- **History storage:** Documents is the only local persistence layer. Do not mirror screening history to Keychain.
