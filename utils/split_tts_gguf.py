#!/usr/bin/env python3
"""
split_tts_gguf.py

Splits the MiniCPM-o TTS GGUF into two files so the transformer portion can be
loaded by standard libllama without error.

Problem:
  The combined GGUF has 193 tensors: 182 standard llama tensors + 11 custom TTS
  tensors (emb_code, emb_text, projector_semantic, projector_spk, head_code).
  Standard libllama's done_getting_tensors() check throws:
    "wrong number of tensors; expected 193, got 182"
  because it only recognises the 182 standard tensors.
  llama.cpp-omni works around this by patching llama-model.cpp to fake-count the
  11 custom tensors. Splitting the file achieves the same result without patching.

Output:
  <stem>-transformer-<type>.gguf  — 182 standard tensors, loadable by libllama
  <stem>-weights-<type>.gguf      — 11 custom TTS tensors, loaded via gguf_init

Usage:
  python3 split_tts_gguf.py <input.gguf> <output_dir>

Example:
  python3 split_tts_gguf.py models/gguf/tts/MiniCPM-o-4_5-tts-F16.gguf models/gguf/tts/
"""

import io
import struct
import sys
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGN   = 32  # tensor data section is 32-byte aligned

# Custom tensors that standard libllama does not recognise for the llama arch.
# These 11 names are hardcoded in llama.cpp-omni/src/llama-model.cpp:2461-2472.
CUSTOM_TENSOR_NAMES = {
    "emb_code.0.weight",
    "emb_text.weight",
    "projector_semantic.linear1.weight",
    "projector_semantic.linear1.bias",
    "projector_semantic.linear2.weight",
    "projector_semantic.linear2.bias",
    "projector_spk.linear1.weight",
    "projector_spk.linear1.bias",
    "projector_spk.linear2.weight",
    "projector_spk.linear2.bias",
    "head_code.0.weight",
}

# GGML type ID → (bytes_per_element, None) for simple types,
#              → (None, (block_elems, bytes_per_block)) for quantised types.
_SIMPLE_TYPES = {0: 4, 1: 2, 30: 2}   # F32, F16, BF16
_BLOCK_TYPES  = {                       # (elements/block, bytes/block)
    2:  (32,  18),   # Q4_0
    3:  (32,  20),   # Q4_1
    6:  (32,  22),   # Q5_0
    7:  (32,  24),   # Q5_1
    8:  (32,  34),   # Q8_0
    10: (256, 116),  # Q2_K
    11: (256, 176),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
}

def tensor_nbytes(dtype: int, dims: list) -> int:
    """Return the exact byte count for a tensor (no alignment padding)."""
    n = 1
    for d in dims:
        n *= d
    if dtype in _SIMPLE_TYPES:
        return n * _SIMPLE_TYPES[dtype]
    if dtype in _BLOCK_TYPES:
        blk_elems, blk_bytes = _BLOCK_TYPES[dtype]
        return (n // blk_elems) * blk_bytes
    raise ValueError(f"Unsupported GGML dtype {dtype} — add it to _SIMPLE_TYPES or _BLOCK_TYPES")

def align_up(n: int, a: int = GGUF_ALIGN) -> int:
    return n + (-n % a)

# ── Low-level binary readers ──────────────────────────────────────────────────

def r_u8 (f): return struct.unpack("<B", f.read(1))[0]
def r_i8 (f): return struct.unpack("<b", f.read(1))[0]
def r_u16(f): return struct.unpack("<H", f.read(2))[0]
def r_i16(f): return struct.unpack("<h", f.read(2))[0]
def r_u32(f): return struct.unpack("<I", f.read(4))[0]
def r_i32(f): return struct.unpack("<i", f.read(4))[0]
def r_f32(f): return struct.unpack("<f", f.read(4))[0]
def r_u64(f): return struct.unpack("<Q", f.read(8))[0]
def r_i64(f): return struct.unpack("<q", f.read(8))[0]
def r_f64(f): return struct.unpack("<d", f.read(8))[0]

def r_str(f) -> bytes:
    return f.read(r_u64(f))

def r_value(f, vtype):
    """Read one GGUF typed value; arrays are returned as (elem_type, count, items)."""
    if vtype == 0:  return r_u8(f)
    if vtype == 1:  return r_i8(f)
    if vtype == 2:  return r_u16(f)
    if vtype == 3:  return r_i16(f)
    if vtype == 4:  return r_u32(f)
    if vtype == 5:  return r_i32(f)
    if vtype == 6:  return r_f32(f)
    if vtype == 7:  return r_u8(f)   # bool stored as uint8
    if vtype == 8:  return r_str(f)
    if vtype == 9:
        et  = r_u32(f)
        cnt = r_u64(f)
        return (et, cnt, [r_value(f, et) for _ in range(cnt)])
    if vtype == 10: return r_u64(f)
    if vtype == 11: return r_i64(f)
    if vtype == 12: return r_f64(f)
    raise ValueError(f"Unknown GGUF vtype {vtype}")

# ── Low-level binary writers ──────────────────────────────────────────────────

def w_u8 (b, v): b.write(struct.pack("<B", v))
def w_i8 (b, v): b.write(struct.pack("<b", v))
def w_u16(b, v): b.write(struct.pack("<H", v))
def w_i16(b, v): b.write(struct.pack("<h", v))
def w_u32(b, v): b.write(struct.pack("<I", v))
def w_i32(b, v): b.write(struct.pack("<i", v))
def w_f32(b, v): b.write(struct.pack("<f", v))
def w_u64(b, v): b.write(struct.pack("<Q", v))
def w_i64(b, v): b.write(struct.pack("<q", v))
def w_f64(b, v): b.write(struct.pack("<d", v))

def w_str(b, s: bytes):
    w_u64(b, len(s))
    b.write(s)

def w_value(b, vtype, v):
    if vtype == 0:  w_u8 (b, v)
    elif vtype == 1:  w_i8 (b, v)
    elif vtype == 2:  w_u16(b, v)
    elif vtype == 3:  w_i16(b, v)
    elif vtype == 4:  w_u32(b, v)
    elif vtype == 5:  w_i32(b, v)
    elif vtype == 6:  w_f32(b, v)
    elif vtype == 7:  w_u8 (b, v)
    elif vtype == 8:  w_str(b, v)
    elif vtype == 9:
        et, cnt, items = v
        w_u32(b, et)
        w_u64(b, cnt)
        for item in items:
            w_value(b, et, item)
    elif vtype == 10: w_u64(b, v)
    elif vtype == 11: w_i64(b, v)
    elif vtype == 12: w_f64(b, v)

# ── GGUF parser ───────────────────────────────────────────────────────────────

def parse_gguf(path: Path):
    """
    Parse a GGUF file header and return (kv_pairs, tensors, data_start_offset).

    kv_pairs : list of (key: bytes, vtype: int, value)
    tensors  : list of dicts with keys: name, n_dims, dims, dtype, offset
    data_start_offset : byte offset in the file where tensor data begins
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == GGUF_MAGIC, f"Not a GGUF file: {path}"
        version   = r_u32(f)
        n_tensors = r_u64(f)
        n_kv      = r_u64(f)

        kv_pairs = []
        for _ in range(n_kv):
            key   = r_str(f)
            vtype = r_u32(f)
            val   = r_value(f, vtype)
            kv_pairs.append((key, vtype, val))

        tensors = []
        for _ in range(n_tensors):
            name   = r_str(f).decode()
            n_dims = r_u32(f)
            dims   = [r_u64(f) for _ in range(n_dims)]
            dtype  = r_u32(f)
            offset = r_u64(f)   # offset relative to data section start
            tensors.append({"name": name, "n_dims": n_dims, "dims": dims,
                             "dtype": dtype, "offset": offset})

        # data section starts at the next 32-byte boundary after the header
        data_start = align_up(f.tell())

    return kv_pairs, tensors, data_start

# ── GGUF writer ───────────────────────────────────────────────────────────────

def write_gguf(out_path: Path, kv_pairs, tensors, src_path: Path, src_data_start: int):
    """
    Write a new GGUF file containing exactly the given subset of tensors.
    Tensor data is read from src_path using the original offsets.
    Offsets in the output file are recomputed from scratch.
    """
    # ── Build header into a BytesIO buffer so we know its size before writing ──
    hdr = io.BytesIO()

    hdr.write(GGUF_MAGIC)
    w_u32(hdr, GGUF_VERSION)
    w_u64(hdr, len(tensors))    # n_tensors
    w_u64(hdr, len(kv_pairs))   # n_kv

    # KV pairs (unchanged from source)
    for key, vtype, val in kv_pairs:
        w_str(hdr, key)
        w_u32(hdr, vtype)
        w_value(hdr, vtype, val)

    # Tensor info with placeholder offsets; record position of each offset
    # field so we can patch them once the real values are known.
    offset_field_positions = []
    for t in tensors:
        w_str(hdr, t["name"].encode())
        w_u32(hdr, t["n_dims"])
        for d in t["dims"]:
            w_u64(hdr, d)
        w_u32(hdr, t["dtype"])
        offset_field_positions.append(hdr.tell())  # position of this offset field
        w_u64(hdr, 0)   # placeholder — patched below

    header_end  = hdr.tell()
    data_start  = align_up(header_end)  # 32-byte-aligned start of data section

    # ── Compute per-tensor data offsets (relative to data_start in output) ──
    out_offsets = []
    cur = 0
    for t in tensors:
        out_offsets.append(cur)
        raw_size = tensor_nbytes(t["dtype"], t["dims"])
        cur      = align_up(cur + raw_size)

    # ── Patch the offset fields in the header buffer ──
    for pos, off in zip(offset_field_positions, out_offsets):
        hdr.seek(pos)
        w_u64(hdr, off)

    # ── Write the file ────────────────────────────────────────────────────────
    with open(out_path, "wb") as dst, open(src_path, "rb") as src:
        dst.write(hdr.getvalue())
        dst.write(b"\x00" * (data_start - header_end))  # alignment padding

        for t, off in zip(tensors, out_offsets):
            raw_size = tensor_nbytes(t["dtype"], t["dims"])

            # Read tensor data from source file
            src.seek(src_data_start + t["offset"])
            data = src.read(raw_size)
            if len(data) != raw_size:
                raise RuntimeError(
                    f"Short read for tensor '{t['name']}': "
                    f"expected {raw_size} bytes, got {len(data)}"
                )

            dst.write(data)
            dst.write(b"\x00" * (align_up(raw_size) - raw_size))  # pad to alignment

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output names: insert "-transformer" / "-weights" before the type
    # suffix (F16, F32, Q4_K_M …).  Falls back to appending if not recognised.
    parts = input_path.stem.rsplit("-", 1)
    TYPE_SUFFIXES = {"F16", "F32", "BF16", "Q4_K_M", "Q4_K_S", "Q8_0", "Q4_0"}
    if len(parts) == 2 and parts[1] in TYPE_SUFFIXES:
        base, suffix = parts
        transformer_name = f"{base}-transformer-{suffix}.gguf"
        weights_name     = f"{base}-weights-{suffix}.gguf"
    else:
        transformer_name = f"{input_path.stem}-transformer.gguf"
        weights_name     = f"{input_path.stem}-weights.gguf"

    transformer_path = output_dir / transformer_name
    weights_path     = output_dir / weights_name

    # ── Parse ─────────────────────────────────────────────────────────────────
    print(f"Parsing {input_path} …")
    kv_pairs, all_tensors, data_start = parse_gguf(input_path)
    print(f"  {len(all_tensors)} tensors, {len(kv_pairs)} KV pairs, "
          f"data section starts at 0x{data_start:x}")

    transformer_tensors = [t for t in all_tensors if t["name"] not in CUSTOM_TENSOR_NAMES]
    weights_tensors     = [t for t in all_tensors if t["name"] in CUSTOM_TENSOR_NAMES]

    found_custom    = {t["name"] for t in weights_tensors}
    missing_custom  = CUSTOM_TENSOR_NAMES - found_custom
    if missing_custom:
        print(f"  WARNING: expected custom tensors not found: {missing_custom}")

    print(f"  → transformer: {len(transformer_tensors)} tensors")
    print(f"  → weights:     {len(weights_tensors)} tensors")
    for t in weights_tensors:
        print(f"      {t['name']}  {t['dims']}  dtype={t['dtype']}")

    # ── Write ─────────────────────────────────────────────────────────────────
    print(f"\nWriting {transformer_path} …")
    write_gguf(transformer_path, kv_pairs, transformer_tensors, input_path, data_start)
    print(f"  {transformer_path.stat().st_size / 1024**2:.1f} MB")

    print(f"Writing {weights_path} …")
    write_gguf(weights_path, kv_pairs, weights_tensors, input_path, data_start)
    print(f"  {weights_path.stat().st_size / 1024**2:.1f} MB")

    print(f"\nDone.")
    print(f"  {transformer_name}  — load with llama_model_load_from_file()")
    print(f"  {weights_name}      — load with gguf_init_from_file()")


if __name__ == "__main__":
    main()
