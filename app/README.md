# CogniTrace iOS App

Flutter app for the CogniTrace on-device Parkinson's voice screening prototype.

Run from this directory:

```bash
flutter pub get
cd ios && pod install && cd ..
flutter run --release
```

Use a physical iPhone running iOS 16.4 or later. The app does not support the
iOS simulator because llamadart and ONNX Runtime need native device frameworks.

See the repository root `README.md` and `docs/DEVELOPMENT.md` for the public
project overview and full setup notes.
