#!/bin/sh
set -eu

min_ios_version="${IPHONEOS_DEPLOYMENT_TARGET:-16.4}"

fix_minimum_os_version() {
  framework_name="$1"
  framework_dir="${TARGET_BUILD_DIR}/${FRAMEWORKS_FOLDER_PATH}/${framework_name}.framework"
  plist_path="${framework_dir}/Info.plist"

  if [ ! -f "$plist_path" ]; then
    return
  fi

  /usr/libexec/PlistBuddy -c "Set :MinimumOSVersion ${min_ios_version}" "$plist_path" 2>/dev/null ||
    /usr/libexec/PlistBuddy -c "Add :MinimumOSVersion string ${min_ios_version}" "$plist_path"

  if [ "${CODE_SIGNING_ALLOWED:-YES}" = "NO" ]; then
    return
  fi

  if [ -n "${EXPANDED_CODE_SIGN_IDENTITY:-}" ]; then
    /usr/bin/codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" \
      --preserve-metadata=identifier,entitlements,flags \
      "$framework_dir"
  fi
}

create_archive_dsym() {
  framework_name="$1"
  binary_path="${TARGET_BUILD_DIR}/${FRAMEWORKS_FOLDER_PATH}/${framework_name}.framework/${framework_name}"
  dsym_root="${DWARF_DSYM_FOLDER_PATH:-}"

  if [ "${ACTION:-}" != "install" ] || [ -z "$dsym_root" ] || [ ! -f "$binary_path" ]; then
    return
  fi

  mkdir -p "$dsym_root"
  /usr/bin/dsymutil "$binary_path" -o "${dsym_root}/${framework_name}.framework.dSYM" || true
}

fix_minimum_os_version "llamadart"
create_archive_dsym "llamadart"
create_archive_dsym "objective_c"
