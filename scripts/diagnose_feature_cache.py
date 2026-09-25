"""Diagnose the frozen-feature cache: completeness, corruption, and phase cost.

Written after `--extract-only` appeared to stop at 15,600/15,699 images. This
reports FACTS rather than accepting a guess:

  1. how many image (.npz) and text (.npy) cache entries exist
  2. whether any .npz is truncated -- an interrupted np.savez leaves a partial
     file that `attach_cached` still keeps, because it only checks existence,
     so the failure would surface later at training time, far from its cause
  3. how long tiny .npy writes take on THIS filesystem
  4. how much batching text encoding buys over one call per phrase

(3) and (4) exist because the two candidate causes for the stall have very
different fixes, and the measured numbers decide between them.

    python scripts/diagnose_feature_cache.py
    python scripts/diagnose_feature_cache.py --bench --checkpoint <remoteclip.pt>

`--cache` may be either layout:

  * a DIRECTORY of one small `.npz` per image (the RemoteCLIP grounding cache), or
  * a single CONSOLIDATED `.npz` file (the fusion-feature cache: `train.npz`).

Exit 0 = a real cache was inspected and reported. Exit 1 = NOTHING was
inspected -- a missing path, a directory holding no `.npz` entries, or a file
that does not load. A diagnostic that opened nothing must not look like a pass,
so the failure paths exit non-zero and say what was expected and what was found.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CACHE = (
    REPO_ROOT / "artifacts" / "grounding" / "remoteclip_grounding_v001"
    / "cache" / "remoteclip_224_v1"
)

DEFAULT_CHECKPOINT = (
    Path.home()
    / ".cache/huggingface/hub/models--chendelong--RemoteCLIP"
    / "snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
)


def hr(t: str = "") -> None:
    print("-" * 68)
    if t:
        print(t)
        print("-" * 68)


def _report_consolidated_file(cache: Path) -> dict:
    """One consolidated `.npz` (the fusion-feature layout: `train.npz`).

    Here the whole split is a single file whose row-oriented arrays share their
    first axis. Globbing it as if it were a directory finds nothing -- which is
    how the original version reported a clean `0` for a 170 MB cache it had not
    opened. The file is opened with `numpy.load` and its real keys, shapes and
    row count are reported.
    """
    size = cache.stat().st_size
    print(f"  cache file           : {cache}")
    print(f"  file bytes           : {size / 1e6:.1f} MB")

    try:
        with np.load(cache) as z:
            keys = list(z.files)
            shapes = {key: tuple(z[key].shape) for key in keys}
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        print(f"  CORRUPT {cache.name}  {detail}")
        return {"ok": False, "reason": "unloadable", "detail": detail}

    if not keys:
        print("  the .npz holds no arrays")
        return {
            "ok": False,
            "reason": "empty_file",
            "detail": "the .npz holds no arrays",
        }

    hr("ARRAYS (consolidated .npz)")
    for key in keys:
        print(f"  {key:<14} {shapes[key]}")

    rows = sorted({shape[0] for shape in shapes.values() if shape})
    entries = rows[0] if len(rows) == 1 else max(rows, default=0)
    print(f"  image entries        : {entries:,}")
    # Only warn when there are two or more DISTINCT row counts; an empty `rows`
    # (a rank-0 scalar with no first axis) has nothing to disagree about, and
    # printing "disagree on their first axis: []" there is meaningless.
    if len(rows) > 1:
        print(f"  WARNING arrays disagree on their first axis: {rows}")

    # A file that holds arrays but no rows opened without yielding anything to
    # inspect -- the same situation as the no-arrays case above and the empty
    # directory, so it fails the verdict too. The printed line above still shows
    # why.
    if entries == 0:
        print("  the .npz holds no rows (0 entries)")
        return {
            "ok": False,
            "reason": "no_rows",
            "detail": "the .npz holds no rows (0 entries)",
        }

    return {
        "ok": True,
        "kind": "file",
        "images": entries,
        "bytes": size,
        "keys": keys,
        "shapes": shapes,
    }


def _report_cache_directory(cache: Path) -> dict:
    """The per-sample layout: a directory of one small `.npz` per image."""
    npz = sorted(cache.glob("*.npz"))
    text_dir = cache / "text"
    npy = sorted(text_dir.glob("*.npy")) if text_dir.exists() else []

    print(f"  cache dir            : {cache}")
    print(f"  image entries (.npz) : {len(npz):,}")
    print(f"  text entries  (.npy) : {len(npy):,}")

    tiny = [p for p in npz if p.stat().st_size < 1024]
    print(f"  npz under 1 KB       : {len(tiny):,}")

    hr("LOADABLE CHECK (sample: first 3 and last 3 by name, deduplicated)")
    bad: list[tuple[str, str]] = []
    # Deduplicate while preserving order: for a directory of 3 or fewer files the
    # two slices overlap, and iterating them raw would sample -- and print -- the
    # same file twice, making a 3-entry cache read as 6.
    sampled = list(dict.fromkeys(npz[:3] + npz[-3:]))
    for p in sampled:
        try:
            with np.load(p) as z:
                shape = z["patches"].shape
            print(f"  OK      {p.name}  patches={shape}")
        except Exception as exc:
            bad.append((p.name, f"{type(exc).__name__}: {exc}"))
            print(f"  CORRUPT {p.name}  {type(exc).__name__}")

    # The sample is a deliberate bound, so name it: a truncation outside the
    # sampled 6 would otherwise read as a clean pass for the whole directory.
    print(f"  {'load-checked':<20} : {len(sampled)} of {len(npz):,} entries "
          "(sampled; the sub-1 KB count above is the whole-directory "
          "truncation signal)")

    if npz:
        hr("NEWEST WRITES (mtime)")
        for p in sorted(npz, key=lambda q: q.stat().st_mtime)[-3:]:
            when = time.strftime("%H:%M:%S", time.localtime(p.stat().st_mtime))
            print(f"  {when}  {p.name}")

    total = sum(p.stat().st_size for p in npz)
    print(f"  image bytes          : {total / 1e6:.1f} MB")

    # A sampled truncation is a real finding: purpose #2 of this tool is to catch
    # a partial .npz before training does, so corruption must fail the verdict
    # even when every other file in the directory is healthy.
    result = {
        "ok": bool(npz) and not bad,
        "kind": "dir",
        "images": len(npz),
        "texts": len(npy),
        "tiny": len(tiny),
        "corrupt_sampled": len(bad),
        "bytes": total,
    }
    if bad:
        result["detail"] = f"{len(bad)} sampled .npz failed to load"
    return result


def report_cache(cache: Path) -> dict:
    """Report what is actually at `cache`, and whether anything was inspected.

    The return value's `ok` is True only when real cache entries were found and
    read. `main` turns a False `ok` into a non-zero exit code, so a path that
    cannot be inspected fails loudly instead of reporting a clean `0`.
    """
    hr("CACHE CONTENTS")
    if not cache.exists():
        print(f"  cache path does not exist: {cache}")
        return {"ok": False, "reason": "missing",
                "detail": "the path does not exist"}
    if cache.is_file():
        return _report_consolidated_file(cache)
    if cache.is_dir():
        return _report_cache_directory(cache)
    print(f"  cache path is neither a file nor a directory: {cache}")
    return {"ok": False, "reason": "unsupported",
            "detail": "the path is neither a file nor a directory"}


def bench_tiny_writes(n: int) -> float:
    hr(f"BENCHMARK: {n} tiny .npy writes (one file per phrase)")
    d = Path(tempfile.mkdtemp(prefix="npy_bench_"))
    vec = np.zeros(512, dtype=np.float16)
    t0 = time.time()
    for i in range(n):
        np.save(d / f"{i:05d}.npy", vec)
    dt = time.time() - t0
    print(f"  {dt:.2f}s for {n} files  ->  {dt / n * 1000:.2f} ms/file")
    print(f"  extrapolated to 20,000 phrases : {dt / n * 20000:.0f}s")
    return dt


def bench_text(phrases: list[str], checkpoint: str) -> None:
    hr("BENCHMARK: text encoding, single vs batched")
    from core.config import load_config
    from specialists.grounding.remoteclip import build_encoder

    cfg = load_config()
    enc = build_encoder(
        cfg, resolution=224, device="cpu", checkpoint_path=checkpoint
    )

    t0 = time.time()
    for p in phrases:
        enc.encode_text([p])
    single = time.time() - t0

    t0 = time.time()
    enc.encode_text(phrases)
    batched = time.time() - t0

    print(f"  {len(phrases)} phrases, one call each : {single:.2f}s")
    print(f"  {len(phrases)} phrases, one batch     : {batched:.2f}s")
    print(f"  speedup                     : {single / max(batched, 1e-6):.1f}x")
    print(f"  extrapolated to 20,000 one-at-a-time : "
          f"{single / max(1, len(phrases)) * 20000:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose the feature cache")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--bench", action="store_true",
                    help="also run the write-cost and text-encoding benchmarks")
    ap.add_argument("--bench-n", type=int, default=300)
    args = ap.parse_args()

    print("=" * 68)
    print("FEATURE CACHE DIAGNOSTIC")
    print("=" * 68)

    cache = Path(args.cache)
    report = report_cache(cache)

    ok = bool(report.get("ok"))
    if not ok:
        hr("DIAGNOSTIC FAILED -- NOTHING WAS INSPECTED")
        print(f"  cache path : {cache}")
        print("  expected   : a consolidated .npz file, or a directory holding "
              "one .npz per image")
        print(f"  found      : {report.get('detail', 'nothing to inspect')}")
        print("  A diagnostic that opened nothing is a FAILURE, not a clean "
              "pass; the exit code below is non-zero.")

    if args.bench:
        bench_tiny_writes(args.bench_n)
        phrases = [f"the object number {i}" for i in range(args.bench_n)]
        cp = Path(args.checkpoint)
        if cp.exists():
            bench_text(phrases, str(cp))
        else:
            hr("TEXT BENCHMARK SKIPPED")
            print(f"  checkpoint not found: {cp}")

    hr("=")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())