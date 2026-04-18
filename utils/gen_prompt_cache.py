#!/usr/bin/env python3
"""
gen_prompt_cache.py

Generate a prompt_cache.gguf for Token2Wav voice cloning from a reference WAV.

The prompt_cache.gguf stores:
  - Speaker embedding (spk_cb)  — controls voice timbre in the flow matching decoder
  - Conformer encoder caches    — key/value caches for streaming inference
  - Estimator (DiT) caches      — per-timestep caches for the flow matching ODE solver

Usage:
  python3 utils/gen_prompt_cache.py <ref_wav> <output.gguf> [options]

Example:
  python3 utils/gen_prompt_cache.py \\
      test_data/alternative.wav \\
      /path/to/alternative_prompt_cache.gguf \\
      --model-path /path/to/token2wav \\
      --n-timesteps 10

The --model-path defaults to the token2wav model directory under models/.

The script monkey-patches torchaudio.load to use soundfile because
the installed torchaudio version requires torchcodec which may not be present (NixOS).
"""

import argparse
import io
import struct
import sys
from pathlib import Path

# ── torchaudio monkey-patch (must happen before any stepaudio2 import) ─────

def _patch_torchaudio_load():
    """Replace torchaudio.load with a soundfile-based implementation."""
    import soundfile as sf
    import torch
    import torchaudio

    def _sf_load(file, *args, **kwargs):
        data, sr = sf.read(str(file), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr

    torchaudio.load = _sf_load

_patch_torchaudio_load()

import numpy as np
import torch


# ── GGUF binary helpers ─────────────────────────────────────────────────────

GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGN   = 32   # tensor data section is 32-byte aligned

# GGUF value types
VTYPE_UINT32 = 4
VTYPE_INT32  = 5
VTYPE_FLOAT32 = 6
VTYPE_STRING  = 8

GGML_TYPE_F32 = 0


def _align_up(n: int, a: int = GGUF_ALIGN) -> int:
    return n + (-n % a)


def _w_u32(b: io.BytesIO, v: int):  b.write(struct.pack("<I", v))
def _w_i32(b: io.BytesIO, v: int):  b.write(struct.pack("<i", v))
def _w_f32(b: io.BytesIO, v: float): b.write(struct.pack("<f", v))
def _w_u64(b: io.BytesIO, v: int):  b.write(struct.pack("<Q", v))


def _w_str(b: io.BytesIO, s: str):
    enc = s.encode("utf-8")
    _w_u64(b, len(enc))
    b.write(enc)


def _w_kv_u32(b: io.BytesIO, key: str, val: int):
    _w_str(b, key)
    _w_u32(b, VTYPE_UINT32)
    _w_u32(b, val)


def _w_kv_i32(b: io.BytesIO, key: str, val: int):
    _w_str(b, key)
    _w_u32(b, VTYPE_INT32)
    _w_i32(b, val)


def _w_kv_f32(b: io.BytesIO, key: str, val: float):
    _w_str(b, key)
    _w_u32(b, VTYPE_FLOAT32)
    _w_f32(b, val)


def _w_tensor_info(b: io.BytesIO, name: str, ne: list[int], dtype: int, offset: int):
    _w_str(b, name)
    _w_u32(b, len(ne))
    for d in ne:
        _w_u64(b, d)
    _w_u32(b, dtype)
    _w_u64(b, offset)


# ── GGML layout conversion ──────────────────────────────────────────────────
#
# GGML stores tensors in "Fortran-like" order: ne[0] is the *innermost*
# (fastest-varying) dimension. PyTorch uses C order: the *last* dim is
# innermost. The conversions below produce GGML-compatible byte arrays.
#
# Naming convention for cache shapes (Python):
#   spk_emb             : [B=1, 192]
#   conformer_cnn_cache : [B=1, C=512, T=6]
#   conformer_att_cache : [depth=10, B=1, nh=8, t, c=128]
#   estimator_cnn_cache : [n_ts_max=16, n_blocks=16, B_cfm=2, C=1024, T=2]
#   estimator_att_cache : [n_ts_max=16, n_blocks=16, B_cfm=2, nh=8, t, c=128]

def _spk_to_ggml(spk_emb: np.ndarray) -> tuple[bytes, list[int]]:
    """spk_emb [1,192] → GGML ne=[192], bytes=192 floats."""
    arr = np.ascontiguousarray(spk_emb.squeeze(0))   # [192]
    return arr.tobytes(), [192]


def _conf_cnn_to_ggml(arr: np.ndarray) -> tuple[bytes, list[int]]:
    """conformer_cnn_cache [1, C, T] → GGML ne=[C, T], C innermost."""
    arr = arr.squeeze(0)               # [C, T]
    C, T = arr.shape
    # PyTorch [C, T] has T innermost; GGML ne=[C, T] needs C innermost.
    # Transpose → [T, C] contiguous → C varies fastest.
    arr = np.ascontiguousarray(arr.T)  # [T, C]
    return arr.tobytes(), [C, T]


def _conf_att_to_ggml(arr: np.ndarray) -> tuple[bytes, list[int]]:
    """conformer_att_cache [depth, B, nh, t, c] → GGML ne=[c, t, nh, depth]."""
    arr = arr.squeeze(1)               # [depth, nh, t, c]  — c already innermost
    depth, nh, t, c = arr.shape
    arr = np.ascontiguousarray(arr)
    return arr.tobytes(), [c, t, nh, depth]


def _est_cnn_to_ggml(arr: np.ndarray, n_timesteps: int) -> tuple[bytes, list[int]]:
    """
    estimator_cnn_cache [n_ts_max, n_blocks, B_cfm, C, T]
    → GGML ne=[C, T, n_ts*n_blocks, B_cfm], C innermost.
    """
    _n_ts_max, n_blocks, B_cfm, C, T = arr.shape
    arr = np.ascontiguousarray(arr[:n_timesteps])   # [n_ts, n_blocks, B_cfm, C, T]
    arr = arr.reshape(n_timesteps * n_blocks, B_cfm, C, T)  # [step, B, C, T]
    # Want [B, step, T, C] with C innermost → C becomes ne[0]
    arr = arr.transpose(1, 0, 3, 2)                         # [B, step, T, C]
    arr = np.ascontiguousarray(arr)
    return arr.tobytes(), [C, T, n_timesteps * n_blocks, B_cfm]


def _est_att_to_ggml(arr: np.ndarray, n_timesteps: int) -> tuple[bytes, list[int]]:
    """
    estimator_att_cache [n_ts_max, n_blocks, B_cfm, nh, t, c]
    → GGML ne=[c, t, n_ts*n_blocks*nh, B_cfm], c innermost.
    """
    _n_ts_max, n_blocks, B_cfm, nh, t, c = arr.shape
    arr = np.ascontiguousarray(arr[:n_timesteps])        # [n_ts, n_blocks, B, nh, t, c]
    arr = arr.reshape(n_timesteps * n_blocks, B_cfm, nh, t, c)  # [step, B, nh, t, c]
    arr = arr.transpose(1, 0, 2, 3, 4)                           # [B, step, nh, t, c]
    arr = np.ascontiguousarray(arr)
    arr = arr.reshape(B_cfm, n_timesteps * n_blocks * nh, t, c)  # [B, combo, t, c]
    arr = np.ascontiguousarray(arr)
    return arr.tobytes(), [c, t, n_timesteps * n_blocks * nh, B_cfm]


# ── GGUF writer ─────────────────────────────────────────────────────────────

def write_prompt_cache_gguf(
    out_path: Path,
    n_timesteps: int,
    temperature: float,
    pre_lookahead: int,
    chunk_main: int,
    chunk_total: int,
    up_rate: int,
    tensors: list[tuple[str, bytes, list[int]]],   # (name, raw_bytes, ggml_ne)
):
    hdr = io.BytesIO()

    hdr.write(GGUF_MAGIC)
    _w_u32(hdr, GGUF_VERSION)
    _w_u64(hdr, len(tensors))   # n_tensors
    _w_u64(hdr, 7)              # n_kv (version, n_timesteps, temperature, pre_lookahead,
                                #        chunk_main, chunk_total, up_rate)

    # KV metadata
    _w_kv_u32(hdr, "mtmd.prompt_cache.version",       2)
    _w_kv_i32(hdr, "mtmd.prompt_cache.n_timesteps",   n_timesteps)
    _w_kv_f32(hdr, "mtmd.prompt_cache.temperature",   temperature)
    _w_kv_i32(hdr, "mtmd.prompt_cache.pre_lookahead", pre_lookahead)
    _w_kv_i32(hdr, "mtmd.prompt_cache.chunk_main",    chunk_main)
    _w_kv_i32(hdr, "mtmd.prompt_cache.chunk_total",   chunk_total)
    _w_kv_i32(hdr, "mtmd.prompt_cache.up_rate",       up_rate)

    # Compute per-tensor data offsets
    offsets = []
    cur = 0
    for _name, raw, ne in tensors:
        offsets.append(cur)
        nbytes = 4 * int(np.prod(ne))   # all F32
        cur = _align_up(cur + nbytes)

    # Tensor info section (offsets patch-written directly)
    for (name, raw, ne), off in zip(tensors, offsets):
        _w_tensor_info(hdr, name, ne, GGML_TYPE_F32, off)

    header_end = hdr.tell()
    data_start = _align_up(header_end)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(hdr.getvalue())
        f.write(b"\x00" * (data_start - header_end))   # alignment padding

        for (name, raw, ne), off in zip(tensors, offsets):
            nbytes = 4 * int(np.prod(ne))
            assert len(raw) == nbytes, f"{name}: expected {nbytes} bytes, got {len(raw)}"
            f.write(raw)
            f.write(b"\x00" * (_align_up(nbytes) - nbytes))   # tensor padding

    print(f"Written {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent.parent.parent   # …/minicpm

    default_model_path = repo_root / "models" / "MiniCPM-o-4_5" / "assets" / "token2wav"
    default_out_dir    = repo_root / "models" / "gguf" / "token2wav-gguf"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_wav",  help="Reference WAV file (16 kHz mono preferred)")
    ap.add_argument("out_gguf", nargs="?", default=None, help="Output .gguf path (default: <out-dir>/<stem>_prompt_cache.gguf)")
    ap.add_argument("--model-path",   default=str(default_model_path), help="Path to token2wav model directory")
    ap.add_argument("--n-timesteps",  type=int,   default=10,  help="Flow matching ODE timesteps (default: 10)")
    ap.add_argument("--temperature",  type=float, default=1.0, help="Flow matching temperature (default: 1.0)")
    args = ap.parse_args()

    ref_wav   = Path(args.ref_wav).resolve()
    model_dir = Path(args.model_path).resolve()

    if args.out_gguf:
        out_path = Path(args.out_gguf).resolve()
    else:
        out_path = default_out_dir / f"{ref_wav.stem}_prompt_cache.gguf"

    if not ref_wav.exists():
        sys.exit(f"ERROR: ref_wav not found: {ref_wav}")
    if not model_dir.exists():
        sys.exit(f"ERROR: model_path not found: {model_dir}")

    print(f"Reference WAV : {ref_wav}")
    print(f"Model path    : {model_dir}")
    print(f"n_timesteps   : {args.n_timesteps}")
    print(f"Output        : {out_path}")
    print()

    # ── Load Token2wav and build caches ─────────────────────────────────────
    print("Loading Token2wav model…")
    from stepaudio2.token2wav import Token2wav
    t2w = Token2wav(str(model_dir), n_timesteps=args.n_timesteps)

    print("Running set_stream_cache…")
    stream_cache, _hift_cache = t2w.set_stream_cache(str(ref_wav))
    spk_emb = t2w.cache[2]   # [1, 192] float32 CUDA

    # Move everything to CPU float32 numpy
    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.cpu().float().numpy()

    spk_np  = to_np(spk_emb)
    cc_np   = to_np(stream_cache["conformer_cnn_cache"])
    ca_np   = to_np(stream_cache["conformer_att_cache"])
    ec_np   = to_np(stream_cache["estimator_cnn_cache"])
    ea_np   = to_np(stream_cache["estimator_att_cache"])

    print(f"  spk_emb             : {spk_np.shape}")
    print(f"  conformer_cnn_cache : {cc_np.shape}")
    print(f"  conformer_att_cache : {ca_np.shape}")
    print(f"  estimator_cnn_cache : {ec_np.shape}  (using [:n_ts={args.n_timesteps}])")
    print(f"  estimator_att_cache : {ea_np.shape}  (using [:n_ts={args.n_timesteps}])")

    # ── Convert to GGML layout ───────────────────────────────────────────────
    print("\nConverting to GGML layout…")
    spk_bytes,  spk_ne  = _spk_to_ggml(spk_np)
    cc_bytes,   cc_ne   = _conf_cnn_to_ggml(cc_np)
    ca_bytes,   ca_ne   = _conf_att_to_ggml(ca_np)
    ec_bytes,   ec_ne   = _est_cnn_to_ggml(ec_np, args.n_timesteps)
    ea_bytes,   ea_ne   = _est_att_to_ggml(ea_np, args.n_timesteps)

    print(f"  prompt_cache.spk_cb               : ne={spk_ne}")
    print(f"  prompt_cache.conformer_cnn_cache  : ne={cc_ne}")
    print(f"  prompt_cache.conformer_att_cache  : ne={ca_ne}")
    print(f"  prompt_cache.estimator_cnn_cache  : ne={ec_ne}")
    print(f"  prompt_cache.estimator_att_cache  : ne={ea_ne}")

    # Architecture constants (from flow.yaml / model design)
    PRE_LOOKAHEAD = 3
    CHUNK_MAIN    = 25
    CHUNK_TOTAL   = 28
    UP_RATE       = 2

    tensors = [
        ("prompt_cache.spk_cb",              spk_bytes, spk_ne),
        ("prompt_cache.conformer_cnn_cache", cc_bytes,  cc_ne),
        ("prompt_cache.conformer_att_cache", ca_bytes,  ca_ne),
        ("prompt_cache.estimator_cnn_cache", ec_bytes,  ec_ne),
        ("prompt_cache.estimator_att_cache", ea_bytes,  ea_ne),
    ]

    print(f"\nWriting GGUF…")
    write_prompt_cache_gguf(
        out_path     = out_path,
        n_timesteps  = args.n_timesteps,
        temperature  = args.temperature,
        pre_lookahead= PRE_LOOKAHEAD,
        chunk_main   = CHUNK_MAIN,
        chunk_total  = CHUNK_TOTAL,
        up_rate      = UP_RATE,
        tensors      = tensors,
    )
    print("Done.")


if __name__ == "__main__":
    main()
