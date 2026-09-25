"""End-to-end check: do the CDVQA annotations and the SECOND imagery actually join?

Why this script exists
----------------------
Two separate things had to become true before a single CDVQA example could be
loaded, and neither was true before this session:

  1. The adapter modelled ONE image per ``file_name``. SECOND ships TWO, and the
     two directories share basenames, so a flat layout physically cannot hold
     them. Fixed by the additive pair-aware change
     (``docs/ARCHITECTURE_CHANGE_CDVQA_IMAGE_PAIRS.md``).
  2. The imagery had never been extracted. ``data/cdvqa/`` held ``annotations/``
     and ``second.zip`` and zero PNGs.

A THIRD defect was found later: ``build_cdvqa_examples`` emitted the flat
``<root>/<file_name>`` for every scene, so all 153,130 examples pointed at a file
that does not exist (0 resolved on disk). This script had passed 9/9 while that
defect was live, because it checked the pair for a *sampled* scene but never the
*examples*. Section [5b] now closes that blind spot: it asserts that EVERY
example's ``image_path`` and every entry of its ``image_paths`` resolve on disk,
and that the corpus is detected as the paired layout.

This script is the check that both are now true *together*, against the real
2,968-scene corpus rather than a synthetic tree. It deliberately asserts the
negative too: the flat ``image_path()`` must NOT resolve, because that is the
honest contrast that proves the pair layout is the one on disk.

Nothing here is inferred. Every count is measured from the filesystem.

Usage:
    python scripts/verify_cdvqa_imagery.py [--root data/cdvqa]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data.cdvqa import (  # noqa: E402
    CDVQA_SPLITS,
    LAYOUT_FLAT,
    LAYOUT_SECOND_PAIRS,
    build_cdvqa_examples,
    load_cdvqa,
    summarize,
)

#: The four directories SECOND's archive ships, in archive order.
ARCHIVE_DIRS = ("im1", "im2", "label1", "label2")

#: How many sampled examples get their pair resolved and byte-compared.
SAMPLE_SIZE = 25


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/cdvqa", help="CDVQA dataset root")
    parser.add_argument(
        "--json",
        default="artifacts/cdvqa/imagery_verification.json",
        help="where to write the evidence artifact",
    )
    args = parser.parse_args()

    root = Path(args.root)
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    print("=" * 72)
    print("CDVQA imagery end-to-end verification")
    print("=" * 72)
    print(f"root: {root.resolve()}")
    print()

    # -- 1. on-disk directory counts ---------------------------------------
    on_disk = {}
    for directory in ARCHIVE_DIRS:
        path = root / directory
        count = sum(1 for _ in path.glob("*.png")) if path.is_dir() else 0
        on_disk[directory] = count
    print("[1] on-disk PNG counts")
    for directory, count in on_disk.items():
        print(f"      {directory:<7} {count:>6,}")
    print(f"      {'total':<7} {sum(on_disk.values()):>6,}")
    check("all four archive dirs hold 2,968 PNGs", all(c == 2968 for c in on_disk.values()), on_disk)
    print()

    # -- 2. the corpus loads WITH imagery required -------------------------
    # Before this session this raised: the paired layout was not recognised and
    # the flat layout was not present.
    print("[2] load with require_images=True (previously raised)")
    try:
        corpus = load_cdvqa(root, splits=CDVQA_SPLITS, require_images=True)
        loaded = True
        load_error = None
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        corpus = None
        loaded = False
        load_error = f"{type(exc).__name__}: {exc}"
    print(f"      loaded: {loaded}" + (f"  error={load_error}" if load_error else ""))
    check("load_cdvqa(require_images=True) succeeds", loaded, load_error or "ok")
    if corpus is None:
        return _finish(checks, args, root, on_disk)
    print()

    # -- 3. per-split declared counts and the union ------------------------
    declared = [image.file_name for image in corpus.images]
    per_split = Counter(image.split for image in corpus.images)
    unique = sorted(set(declared))
    print("[3] declared images")
    for split in CDVQA_SPLITS:
        print(f"      {split:<6} {per_split.get(split, 0):>6,} rows")
    print(f"      unique {len(unique):>6,}  (Test/Test2 share imagery by design)")
    check("unique declared file_names == 2,968", len(unique) == 2968, len(unique))
    print()

    # -- 4. every declared name resolves in im1 AND im2 --------------------
    missing = {directory: [] for directory in ARCHIVE_DIRS}
    for name in unique:
        for directory in ARCHIVE_DIRS:
            if not (root / directory / name).exists():
                missing[directory].append(name)
    print("[4] declared file_names present in each directory")
    for directory in ARCHIVE_DIRS:
        print(f"      {directory:<7} missing={len(missing[directory]):>4,}")
    check(
        "every declared name exists in im1 and im2 and both label dirs",
        all(not v for v in missing.values()),
        {k: len(v) for k, v in missing.items()},
    )
    print()

    # -- 5. examples build, and a sample resolves to two REAL files --------
    examples = build_cdvqa_examples(corpus.images, corpus.questions, image_dir=root)
    print("[5] examples")
    print(f"      built: {len(examples):,}")
    check("examples build against the real root", len(examples) > 0, len(examples))

    step = max(1, len(corpus.images) // SAMPLE_SIZE)
    sampled = corpus.images[::step][:SAMPLE_SIZE]
    pair_missing = []
    identical_bytes = []
    flat_resolved = 0
    for image in sampled:
        first, second = image.image_paths(root, layout=LAYOUT_SECOND_PAIRS)
        if not (first.exists() and second.exists()):
            pair_missing.append(image.file_name)
            continue
        if first.read_bytes() == second.read_bytes():
            identical_bytes.append(image.file_name)
        # The flat path is expected NOT to exist -- that is the contrast.
        if image.image_path(root).exists():
            flat_resolved += 1
    print(f"      sampled {len(sampled)} scenes")
    print(f"      pairs missing        : {len(pair_missing)}")
    print(f"      im1 == im2 (bytes)   : {len(identical_bytes)}")
    print(f"      flat image_path() ok : {flat_resolved}  (expected 0)")
    check("sampled pairs both exist on disk", not pair_missing, pair_missing[:5])
    check(
        "sampled im1/im2 differ in content (different acquisitions)",
        not identical_bytes,
        identical_bytes[:5],
    )
    check(
        "flat image_path() does NOT resolve (pair layout is what is on disk)",
        flat_resolved == 0,
        flat_resolved,
    )
    print()

    # -- 5b. every example's paths resolve (the defect-#9 regression guard) --
    # This script passed 9/9 while defect #9 was live, because it never checked
    # the EXAMPLE paths -- only the pair for a sampled scene. A consumer reads
    # example["image_path"], and before the fix that was the flat path, which
    # does not exist: 0 / 153,130 examples resolved. Check every example.
    example_count = len(examples)
    example_layout = examples[0]["layout"] if examples else None
    path_resolved = sum(1 for e in examples if Path(e["image_path"]).is_file())
    pairs_resolved = sum(
        1 for e in examples if all(Path(p).is_file() for p in e["image_paths"])
    )
    print("[5b] example paths resolve (defect #9 guard)")
    print(f"      examples            : {example_count:,}")
    print(f"      layout              : {example_layout}")
    print(f"      image_path resolves : {path_resolved:,}/{example_count:,}")
    print(f"      image_paths resolve : {pairs_resolved:,}/{example_count:,}")
    check(
        "every example image_path resolves on disk",
        example_count > 0 and path_resolved == example_count,
        {"resolved": path_resolved, "total": example_count},
    )
    check(
        "every example image_paths entry resolves on disk",
        example_count > 0 and pairs_resolved == example_count,
        {"resolved": pairs_resolved, "total": example_count},
    )
    check(
        "examples use LAYOUT_SECOND_PAIRS on the real corpus",
        example_layout == LAYOUT_SECOND_PAIRS,
        example_layout,
    )
    print()

    # -- 6. summarize sees imagery present ---------------------------------
    summary = summarize(corpus.images, corpus.questions, examples=examples, image_root=root)
    present = summary.get("image_files_present")
    declared_n = summary.get("image_files_declared")
    print("[6] summarize presence")
    print(f"      declared: {declared_n}")
    print(f"      present : {present}")
    check(
        "summarize reports imagery present, not absent",
        present is not None and present == declared_n,
        {"declared": declared_n, "present": present},
    )
    print()

    return _finish(
        checks,
        args,
        root,
        on_disk,
        extra={
            "summary_presence": {"declared": declared_n, "present": present},
            "examples": {
                "count": example_count,
                "layout": example_layout,
                "image_path_resolved": path_resolved,
                "image_paths_resolved": pairs_resolved,
            },
        },
    )


def _finish(checks, args, root, on_disk, extra=None):
    failures = [c for c in checks if not c["passed"]]
    print("=" * 72)
    print(f"RESULT: {len(checks) - len(failures)}/{len(checks)} checks passed")
    if failures:
        for c in failures:
            print(f"  FAILED: {c['check']}  -> {c['detail']}")
    print("=" * 72)

    payload = {
        "phase": "10",
        "kind": "cdvqa_imagery_verification",
        "root": str(root.resolve()),
        "archive_dirs_on_disk": on_disk,
        "checks": checks,
        "all_checks_passed": not failures,
        "temporal_order_claimed": False,
        "temporal_order_evidence": "artifacts/cdvqa/temporal_order_evidence_v2.json",
        "note": (
            "im1/im2 are returned in ARCHIVE order by this script, which does not "
            "itself establish the temporal order. The label order (label1 = PRE, "
            "label2 = POST) is established elsewhere from the annotations, and the "
            "image order (im1 = PRE, im2 = POST) is supported by a content test — "
            "supported, not proven — see "
            "artifacts/cdvqa/temporal_order_evidence_v2.json."
        ),
    }
    if extra:
        payload.update(extra)

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"artifact: {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
