#!/usr/bin/env python3
"""Merge LoRA adapter into base Gemma and export GGUF with tokenizer patch.

Two-stage export to avoid Unsloth's bundled converter producing corrupted
sliding_window_pattern metadata for Gemma 4:

  Stage 1: Unsloth merges LoRA into 16-bit HF safetensors
  Stage 2: llama.cpp convert_hf_to_gguf.py + llama-quantize (correct metadata)

This fixes the Metal GPU OOM on iPhone caused by llama.cpp misidentifying
SWA vs global-attention layers from bad GGUF metadata.

Requires:
  - llama.cpp repo cloned (set LLAMA_CPP_DIR env var or --llama-cpp flag)
  - llama-quantize built (cmake --build build --target llama-quantize)

Usage:
    python export_gguf.py
    python export_gguf.py --lora outputs/lora_model --out outputs/gguf
    python export_gguf.py --lora outputs/lora_model --out outputs/gguf --skip-patch
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys

LORA_DIR = os.path.join(os.path.dirname(__file__), "outputs", "lora_model")
GGUF_DIR = os.path.join(os.path.dirname(__file__), "outputs", "gguf")
QUANT_METHOD = "q4_k_m"

# Gemma special tokens that must be CONTROL type (not NORMAL)
SPECIAL_TOKENS = ["<start_of_turn>", "<end_of_turn>"]

# GGUF token type values
GGUF_TOKEN_TYPE_NORMAL = 1
GGUF_TOKEN_TYPE_CONTROL = 3

# Default llama.cpp directory (override with --llama-cpp or LLAMA_CPP_DIR env)
DEFAULT_LLAMA_CPP_DIR = os.path.join(os.path.dirname(__file__), "..", "llama.cpp")


def load_and_merge_16bit(lora_dir: str, out_dir: str):
    """Load LoRA adapter, merge into base model at 16-bit, save HF safetensors.

    Loads the base model in BF16 (not 4-bit) to avoid dequantization noise
    that inflates GGUF size when re-quantized.
    """
    import torch
    from unsloth import FastLanguageModel

    print(f"Loading LoRA adapter from: {lora_dir}")

    # Load in 16-bit for clean merge (no NF4 dequant noise)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_dir,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    merged_dir = os.path.join(out_dir, "merged_16bit")
    os.makedirs(merged_dir, exist_ok=True)
    print(f"Saving merged 16-bit model to: {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged 16-bit checkpoint saved ({_dir_size_mb(merged_dir):.0f} MB)")
    return merged_dir


def convert_hf_to_gguf(merged_dir: str, out_dir: str, llama_cpp_dir: str):
    """Convert HF safetensors to F16 GGUF using llama.cpp's converter.

    This produces correct Gemma 4 metadata including sliding_window_pattern
    as bool[] (not uint32[]), which is required for proper iSWA KV cache
    allocation on Metal.
    """
    converter = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.isfile(converter):
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at: {converter}\n"
            f"Clone llama.cpp: git clone https://github.com/ggml-org/llama.cpp\n"
            f"Then set --llama-cpp or LLAMA_CPP_DIR env var."
        )

    f16_path = os.path.join(out_dir, "model-f16.gguf")
    print(f"Converting HF -> F16 GGUF: {f16_path}")

    result = subprocess.run(
        [
            sys.executable, converter, merged_dir,
            "--outfile", f16_path,
            "--outtype", "f16",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}", file=sys.stderr)
        raise RuntimeError(f"convert_hf_to_gguf.py failed (exit {result.returncode})")

    size_mb = os.path.getsize(f16_path) / (1024 * 1024)
    print(f"F16 GGUF written: {f16_path} ({size_mb:.1f} MB)")
    return f16_path


def quantize_gguf(f16_path: str, out_dir: str, quant_method: str, llama_cpp_dir: str):
    """Quantize F16 GGUF to target quantization using llama-quantize."""
    quantize_bin = _find_quantize_binary(llama_cpp_dir)

    base_name = os.path.splitext(os.path.basename(f16_path))[0].replace("-f16", "")
    quant_filename = f"{base_name}-{quant_method.upper().replace('_', '_')}.gguf"
    quant_path = os.path.join(out_dir, quant_filename)

    print(f"Quantizing F16 -> {quant_method}: {quant_path}")

    result = subprocess.run(
        [quantize_bin, f16_path, quant_path, quant_method.upper()],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}", file=sys.stderr)
        raise RuntimeError(f"llama-quantize failed (exit {result.returncode})")

    size_mb = os.path.getsize(quant_path) / (1024 * 1024)
    print(f"Quantized GGUF written: {quant_path} ({size_mb:.1f} MB)")
    return quant_path


def verify_gguf_metadata(gguf_path: str):
    """Verify critical Gemma 4 metadata in the GGUF file.

    Checks that sliding_window_pattern is present and correctly typed,
    which is required for proper iSWA dual-cache allocation in llama.cpp.
    """
    print(f"\nVerifying GGUF metadata: {gguf_path}")

    try:
        from gguf import GGUFReader
    except ImportError:
        print("WARNING: gguf package not installed. Skipping metadata verification.")
        print("  Install: pip install gguf")
        return True

    reader = GGUFReader(gguf_path)

    # Check architecture
    arch = None
    swa_pattern = None
    swa_size = None
    n_layer = None

    for field in reader.fields.values():
        name = field.name
        if name == "general.architecture":
            arch = str(field.parts[field.data[0]], "utf-8")
        elif "sliding_window_pattern" in name:
            swa_pattern = field
        elif "sliding_window" in name and "pattern" not in name:
            swa_size = field.parts[field.data[0]]
        elif name.endswith(".block_count"):
            n_layer = field.parts[field.data[0]]

    ok = True

    if arch:
        print(f"  Architecture: {arch}")
    else:
        print("  WARNING: No architecture field found")
        ok = False

    if n_layer is not None:
        print(f"  Layers: {n_layer}")

    if swa_size is not None:
        print(f"  Sliding window size: {swa_size}")

    if swa_pattern is not None:
        arr_len = len(swa_pattern.data)
        print(f"  Sliding window pattern: {arr_len} entries (expected {n_layer or '?'})")
        if n_layer is not None and arr_len != n_layer:
            print(f"  ERROR: Pattern length {arr_len} != layer count {n_layer}")
            ok = False
        else:
            print(f"  Sliding window pattern: OK")
    else:
        print("  WARNING: No sliding_window_pattern found. llama.cpp may misallocate KV cache.")
        ok = False

    # Check tensor count for output.weight duplication
    tensor_names = [t.name for t in reader.tensors]
    has_token_embd = any("token_embd" in n for n in tensor_names)
    has_output = any(n == "output.weight" for n in tensor_names)
    if has_token_embd and has_output:
        print("  WARNING: Both token_embd.weight and output.weight present (not tied).")
        print("  This uses ~500MB extra GPU memory on E2B.")
    elif has_token_embd:
        print("  Embeddings: tied (token_embd only) - OK")

    print(f"  Total tensors: {len(tensor_names)}")

    if ok:
        print("  Metadata verification: PASSED")
    else:
        print("  Metadata verification: FAILED (see warnings above)")
    return ok


def patch_tokenizer_types(gguf_path: str, special_tokens: list[str]) -> bool:
    """Fix start_of_turn/end_of_turn token types from NORMAL to CONTROL.

    Unsloth may export these as NORMAL (1) instead of CONTROL (3).
    If the bug is fixed upstream this patch is a no-op (tokens already CONTROL).
    Returns True if any tokens were patched, False if all were already correct.

    This is a binary patch on the raw GGUF file. It locates the token vocabulary
    array and the token type array, then updates the type entries for matched tokens.
    """
    print(f"Scanning tokenizer in: {gguf_path}")

    with open(gguf_path, "rb") as f:
        data = bytearray(f.read())

    # GGUF magic: "GGUF" at offset 0
    if data[:4] != b"GGUF":
        print("WARNING: File does not start with GGUF magic. Skipping patch.")
        return False

    # Locate the tokenizer.ggml.tokens string array and tokenizer.ggml.token_type array.
    # Strategy: find each special token as a length-prefixed UTF-8 string (uint64 len + bytes),
    # then locate the type array via the metadata key "tokenizer.ggml.token_type".
    # Because full GGUF parsing is complex we use a targeted search approach:
    # find the token type array key, walk its uint32 values, and patch by index.

    tokens_key = b"tokenizer.ggml.tokens"
    types_key = b"tokenizer.ggml.token_type"

    tokens_key_pos = data.find(tokens_key)
    types_key_pos = data.find(types_key)

    if tokens_key_pos == -1 or types_key_pos == -1:
        print("WARNING: Could not locate tokenizer metadata keys. Skipping patch.")
        return False

    # Build token index from the tokens array.
    # Each string in a GGUF string array: uint64 length + UTF-8 bytes.
    # The array is preceded by its element count (uint64).
    # We search for the count field by scanning after the key + value type bytes.
    token_index = _build_token_index(data, tokens_key_pos, special_tokens)
    if not token_index:
        print("Special tokens not found in vocabulary. Patch not needed or vocab not parseable.")
        return False

    found_names = set(token_index.keys())
    expected_names = set(special_tokens)
    if found_names != expected_names:
        missing = expected_names - found_names
        print(f"WARNING: Only found {found_names}, missing {missing}. Aborting patch to avoid corrupting wrong entries.")
        return False
    for name, idx in token_index.items():
        assert idx >= 0, f"Negative token index for {name}: {idx}"
        assert idx < 1_000_000, f"Implausibly large token index for {name}: {idx}"
    print(f"Token index verified: {token_index}")

    patched = _patch_type_array(data, types_key_pos, token_index)
    if not patched:
        print("Token types already correct. No patch needed.")
        return False

    with open(gguf_path, "wb") as f:
        f.write(data)

    print(f"Patched {len(token_index)} token type(s): {list(token_index.keys())}")
    return True


def _build_token_index(data: bytearray, key_pos: int, targets: list[str]) -> dict[str, int]:
    """Return {token_string: array_index} for each target found in the token vocab array."""
    target_bytes = {t.encode("utf-8") for t in targets}
    result: dict[str, int] = {}

    # Scan forward from the key for length-prefixed strings.
    # We do a simple scan: look for each target as a uint64-prefixed occurrence.
    pos = key_pos
    end = len(data)
    idx = 0

    while pos < end - 8:
        length = struct.unpack_from("<Q", data, pos)[0]
        if length == 0 or length > 512:
            pos += 1
            continue
        if pos + 8 + length > end:
            pos += 1
            continue
        candidate = bytes(data[pos + 8 : pos + 8 + length])
        if candidate in target_bytes:
            result[candidate.decode("utf-8")] = idx
        pos += 8 + length
        idx += 1
        if idx > 1_000_000:
            break

    return result


def _patch_type_array(data: bytearray, types_key_pos: int, token_index: dict[str, int]) -> bool:
    """Locate the token type uint32 array and set entries for indexed tokens to CONTROL."""
    # After the key string there is a value type field (uint32 = 9 for array)
    # then an element type (uint32 = 5 for uint32) and element count (uint64).
    # Each element is a uint32.
    pos = types_key_pos + len(b"tokenizer.ggml.token_type")
    if pos + 16 > len(data):
        return False

    # Skip the key's length prefix that precedes it in the file
    # We need to find where the uint32 values start. Use a heuristic:
    # scan forward for a run of 4-byte values that plausibly matches vocab size.
    # Look for the array start marker: value_type=9 (array), elem_type=5 (u32).
    patched = False
    search_end = min(pos + 256, len(data) - 16)
    while pos < search_end:
        value_type = struct.unpack_from("<I", data, pos)[0]
        if value_type == 9:  # GGUF_TYPE_ARRAY
            elem_type = struct.unpack_from("<I", data, pos + 4)[0]
            if elem_type == 5:  # GGUF_TYPE_UINT32
                count = struct.unpack_from("<Q", data, pos + 8)[0]
                array_start = pos + 16
                if count < 1_000_000 and array_start + count * 4 <= len(data):
                    for token_str, idx in token_index.items():
                        if idx < count:
                            entry_pos = array_start + idx * 4
                            current = struct.unpack_from("<I", data, entry_pos)[0]
                            if current != GGUF_TOKEN_TYPE_CONTROL:
                                struct.pack_into("<I", data, entry_pos, GGUF_TOKEN_TYPE_CONTROL)
                                print(f"  {token_str}: type {current} -> {GGUF_TOKEN_TYPE_CONTROL}")
                                patched = True
                            else:
                                print(f"  {token_str}: already CONTROL (type {current}), no change")
                    return patched
        pos += 1
    return False


def validate_with_llama_cli(gguf_path: str, llama_cpp_dir: str):
    """Run a quick sanity check with llama-cli if available."""
    cli = shutil.which("llama-cli") or shutil.which("llama")
    if not cli:
        # Try the build directory
        candidate = os.path.join(llama_cpp_dir, "build", "bin", "llama-cli")
        if os.path.isfile(candidate):
            cli = candidate
    if not cli:
        print("llama-cli not found. Skipping validation (build llama.cpp to enable).")
        return

    print(f"Validating with: {cli}")
    result = subprocess.run(
        [cli, "-m", gguf_path, "-p", "Hello", "-n", "8", "--log-disable"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        print("llama-cli validation: OK")
    else:
        print(f"llama-cli validation: FAILED (exit {result.returncode})")
        print(result.stderr[:500])


def _find_quantize_binary(llama_cpp_dir: str) -> str:
    """Locate the llama-quantize binary."""
    # Check PATH first
    path_bin = shutil.which("llama-quantize")
    if path_bin:
        return path_bin

    # Check llama.cpp build directory
    candidates = [
        os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize"),
        os.path.join(llama_cpp_dir, "llama-quantize"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        f"llama-quantize not found in PATH or {llama_cpp_dir}/build/bin/\n"
        f"Build it: cd {llama_cpp_dir} && cmake -B build && cmake --build build --target llama-quantize"
    )


def _dir_size_mb(path: str) -> float:
    """Total size of all files in a directory, in MB."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA and export Gemma GGUF")
    parser.add_argument("--lora", default=LORA_DIR, help="Path to saved LoRA adapter")
    parser.add_argument("--out", default=GGUF_DIR, help="Output directory for GGUF")
    parser.add_argument("--quant", default=QUANT_METHOD, help="Quantization method")
    parser.add_argument("--skip-patch", action="store_true", help="Skip tokenizer type patch")
    parser.add_argument("--skip-validate", action="store_true", help="Skip llama-cli validation")
    parser.add_argument(
        "--llama-cpp",
        default=os.environ.get("LLAMA_CPP_DIR", DEFAULT_LLAMA_CPP_DIR),
        help="Path to llama.cpp repo (default: LLAMA_CPP_DIR env or ../llama.cpp)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.lora):
        print(f"LoRA directory not found: {args.lora}", file=sys.stderr)
        print("Run finetune_gemma_qlora.py first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    # Stage 1: Merge LoRA into 16-bit HF checkpoint
    merged_dir = load_and_merge_16bit(args.lora, args.out)

    # Stage 2: Convert HF -> F16 GGUF using llama.cpp's converter
    f16_path = convert_hf_to_gguf(merged_dir, args.out, args.llama_cpp)

    # Stage 3: Quantize F16 -> target quantization
    quant_path = quantize_gguf(f16_path, args.out, args.quant, args.llama_cpp)

    # Stage 4: Patch tokenizer if needed
    if not args.skip_patch:
        patch_tokenizer_types(quant_path, SPECIAL_TOKENS)
    else:
        print("Tokenizer patch skipped (--skip-patch).")

    # Stage 5: Verify GGUF metadata
    verify_gguf_metadata(quant_path)

    # Stage 6: Validate with llama-cli
    if not args.skip_validate:
        validate_with_llama_cli(quant_path, args.llama_cpp)

    # Cleanup: remove intermediate F16 GGUF (large)
    if os.path.isfile(f16_path):
        f16_size = os.path.getsize(f16_path) / (1024 * 1024)
        print(f"\nRemoving intermediate F16 GGUF ({f16_size:.0f} MB): {f16_path}")
        os.remove(f16_path)

    print(f"\nExport complete: {quant_path}")
    print(f"Size: {os.path.getsize(quant_path) / (1024 * 1024):.1f} MB")
    print(f"\nUpload to HuggingFace:")
    print(f"  huggingface-cli upload littlebull9/cognitrace-gemma4-medical-GGUF {quant_path}")


if __name__ == "__main__":
    main()
