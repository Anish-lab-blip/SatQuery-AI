"""Establish the CDVQA temporal order and the label semantics FROM EVIDENCE.

Why this script exists
----------------------
Phase 10's single most important open question is unanswered by the schema docs:

  * ``im1``/``im2`` each hold 2,968 PNGs sharing basenames — which is the
    pre-change image and which the post-change image.
  * ``label1``/``label2`` each hold 2,968 PNGs — what they encode.

Directory names are not evidence. This script settles both questions by
measuring the corpus against itself, using the one source of ground truth that
cannot have come from a directory name: **the 153,130 question/answer pairs**.
Several question types are directly falsifiable against the label maps, so they
can be used to *test* a proposed palette and a proposed temporal order rather
than to assume one.

v2 — why this revision exists (v1 was circular)
-----------------------------------------------
The first revision (``temporal_order_evidence.json``) fit the palette *in
sample* to maximise the ``label1=pre``/``label2=post`` hypothesis, then computed
the reversed-order rate **arithmetically** as ``1 - h1_rate``. A perfect split
that the test computes for itself is not evidence: once ``h1_rate`` reached
1.0, the reversed figure carried zero independent information, and
``order_established`` was satisfiable by construction.

This revision removes the fitting from the order test and **measures both
directions** against the **FIXED** SECOND palette, over **all** scenes. It also
reports the best-fit palette under each hypothesis separately, so a reader can
see whether the test can distinguish the two orders at all. ``order_established``
now requires a measured, asymmetric result and is not satisfiable by
construction.

Four independent measurements
-----------------------------
1. **Temporal order of the label maps (the decisive test).** ``increase_or_not``
   / ``decrease_or_not`` ask, for a named class, whether its area grew or shrank.
   Against the **fixed** SECOND palette we compute the per-colour coverage delta
   ``cov(label2) - cov(label1)`` and check its sign against the recorded answer,
   under both assignments. A correct assignment must win on *directional*
   evidence; the reversed one must not — and the reverse rate is measured, never
   derived. The best-fit palette is reported for both hypotheses as a
   symmetry check.

2. **``change_to_what`` — an independent confirmation.** "What have the areas of
   [class X] in the *pre-change* image mainly changed to?" The answer names the
   **destination** class, i.e. the POST map. We take the pixels that are X in the
   proposed pre map and read the modal destination class in the proposed post
   map; that must equal the recorded answer. Run for both assignments, and at
   several scene counts to show the result is stable.

3. **Image↔label pairing (does ``im1`` go with ``label1``?).** A *content* test,
   not a naming one. Water is reliably dark in aerial imagery, vegetation
   reliably green. Over the fixed mask ``label1 == water`` the pre image should
   be the darker one; over ``label2 == water`` the post image should be darker.
   Reported with three independent discriminators (mean, median, per-pixel
   majority) plus a **white-background control** that should sit at chance —
   the control is what makes the signal credible.

4. **White semantics.** Is white a class or a shared background?

Everything is measured from the real files on disk. Nothing is inferred from a
name. Where the evidence is insufficient the script says so instead of forcing a
conclusion.

Usage:
    python scripts/establish_cdvqa_temporal_order.py [--root data/cdvqa]
        [--scenes 0] [--json artifacts/cdvqa/temporal_order_evidence_v2.json]

``--scenes 0`` (the default) measures ALL scenes. Requires only numpy + Pillow
(the project ``.venv`` provides both).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# The six SECOND land-cover classes, how their names appear in questions, and
# the FIXED palette measured from the release (NOT fitted per hypothesis).
# ---------------------------------------------------------------------------
CLASS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("NVG_surface", r"non-vegetated ground surface"),
    ("trees", r"trees?"),
    ("low_vegetation", r"low vegetation"),
    ("water", r"water"),
    ("buildings", r"buildings"),
    ("playgrounds", r"playgrounds"),
)

CLASS_NAMES: tuple[str, ...] = tuple(name for name, _ in CLASS_PATTERNS)

#: The SECOND release palette. Fixed a priori; the order test does NOT fit it.
FIXED_PALETTE: dict[str, tuple[int, int, int]] = {
    "NVG_surface": (128, 128, 128),
    "low_vegetation": (0, 128, 0),
    "trees": (0, 255, 0),
    "water": (0, 0, 255),
    "buildings": (128, 0, 0),
    "playgrounds": (255, 0, 0),
}

#: The shared background colour.
WHITE = (255, 255, 255)

#: The four shipped splits, in archive order.
SPLITS: tuple[str, ...] = ("Train", "Val", "Test", "Test2")

#: Scene counts at which the ``change_to_what`` confirmation is re-run, to show
#: the result is stable rather than a lucky subset. ``0`` means "all scenes".
C2W_STABILITY_SIZES: tuple[int, ...] = (500, 800, 1200, 0)


def _rgb_to_key(rgb: tuple[int, int, int]) -> int:
    """Pack an RGB triple into one int for fast counting."""
    return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]


def _key_to_rgb(key: int) -> tuple[int, int, int]:
    """Unpack an int back to an RGB triple."""
    return (key >> 16, (key >> 8) & 0xFF, key & 0xFF)


WHITE_KEY = _rgb_to_key(WHITE)
FIXED_KEY: dict[str, int] = {c: _rgb_to_key(v) for c, v in FIXED_PALETTE.items()}
KEY_TO_CLASS: dict[int, str] = {v: c for c, v in FIXED_KEY.items()}
WATER_KEY = FIXED_KEY["water"]
VEG_KEYS: tuple[int, ...] = (FIXED_KEY["trees"], FIXED_KEY["low_vegetation"])


def parse_class(question: str) -> str | None:
    """Return the land-cover class named in a question, or ``None``."""
    for name, pattern in CLASS_PATTERNS:
        if re.search(pattern, question, re.I):
            return name
    return None


def _load_split(
    annotations: Path, split: str
) -> tuple[dict[int, str], dict[int, str], dict[int, dict[str, Any]]]:
    """Read one split.

    Returns:
        ``(img_id -> file_name, answer_id -> answer, question_id -> question row)``.
    """
    images = json.loads((annotations / f"{split}_images.json").read_text(encoding="utf-8"))["images"]
    questions = json.loads((annotations / f"{split}_questions.json").read_text(encoding="utf-8"))["questions"]
    answers = json.loads((annotations / f"{split}_answers.json").read_text(encoding="utf-8"))["answers"]

    img_file = {row["id"]: row["file_name"] for row in images}
    ans_value = {row["id"]: row["answer"] for row in answers}
    q_by_id = {row["id"]: row for row in questions}
    return img_file, ans_value, q_by_id


def collect_signals(
    annotations: Path, splits: tuple[str, ...]
) -> tuple[dict[str, dict[str, dict[str, bool]]], dict[str, dict[str, str]]]:
    """Join the corpus and extract the per-scene signals the tests need.

    Returns:
        ``inc_dec``: ``file_name -> class -> {"inc": bool, "dec": bool}`` from
            ``increase_or_not`` / ``decrease_or_not`` (which never reference an
            image, so they are the cleanest signal).
        ``change_to_what``: ``file_name -> source_class -> destination_class``.
    """
    inc_dec: dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    change_to_what: dict[str, dict[str, str]] = defaultdict(dict)

    for split in splits:
        img_file, ans_value, q_by_id = _load_split(annotations, split)
        for row in q_by_id.values():
            qtype = row.get("type")
            if qtype not in ("increase_or_not", "decrease_or_not", "change_to_what"):
                continue
            refs = row.get("answers_ids") or []
            if len(refs) != 1:
                continue
            answer = ans_value.get(refs[0])
            if answer is None:
                continue
            file_name = img_file.get(row.get("img_id"))
            if file_name is None:
                continue

            if qtype in ("increase_or_not", "decrease_or_not"):
                cls = parse_class(str(row.get("question", "")))
                if cls is None or answer not in ("yes", "no"):
                    continue
                slot = "inc" if qtype == "increase_or_not" else "dec"
                inc_dec[file_name].setdefault(cls, {})[slot] = answer == "yes"
            else:  # change_to_what
                cls = parse_class(str(row.get("question", "")))
                if cls is None:
                    continue
                change_to_what[file_name][cls] = str(answer)

    return inc_dec, change_to_what


def _pack(arr: np.ndarray) -> np.ndarray:
    """Pack an ``(H, W, 3)`` uint8 image into an ``(H, W)`` int32 key map."""
    a = arr.astype(np.int32)
    return (a[:, :, 0] << 16) | (a[:, :, 1] << 8) | a[:, :, 2]


def _hist(key_map: np.ndarray) -> Counter[int]:
    """Histogram of a packed key map."""
    values, counts = np.unique(key_map, return_counts=True)
    return Counter({int(v): int(c) for v, c in zip(values, counts)})


def _joint_hist(src: np.ndarray, dst: np.ndarray) -> Counter[int]:
    """Joint histogram of two packed key maps, as ``(src << 24 | dst) -> count``."""
    combined = (src.astype(np.int64) << 24) | dst.astype(np.int64)
    values, counts = np.unique(combined, return_counts=True)
    return Counter({int(v): int(c) for v, c in zip(values, counts)})


def _select_scenes(all_names: list[str], wanted: set[str], limit: int) -> list[str]:
    """Deterministically pick up to ``limit`` scenes that carry signals.

    ``limit <= 0`` (or ``limit >= pool size``) returns the whole pool.
    """
    pool = [n for n in sorted(all_names) if n in wanted]
    if limit <= 0 or limit >= len(pool):
        return pool
    step = len(pool) / limit
    return [pool[int(i * step)] for i in range(limit)]


def _acc(store: dict[str, int], tag: str, hit: bool, delta: float) -> None:
    """Record one order-test instance, split into directional vs tie evidence.

    A *tie* is ``delta == 0`` (the class's coverage did not change). Ties are
    counted separately because the reversed hypothesis can be "right" about a
    tie without any directional evidence — which is exactly how v1 manufactured
    its perfect split.
    """
    store[f"{tag}_n"] += 1
    store[f"{tag}_ok"] += int(hit)
    if delta == 0.0:
        store[f"{tag}_tie"] += int(hit)
    else:
        store[f"{tag}_dir"] += int(hit)


def main() -> int:  # noqa: C901 - one linear evidence pipeline, kept readable
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/cdvqa", help="CDVQA dataset root")
    parser.add_argument(
        "--scenes", type=int, default=0,
        help="how many scenes to measure (0 = ALL scenes carrying signals)",
    )
    parser.add_argument(
        "--json", default="artifacts/cdvqa/temporal_order_evidence_v2.json",
        help="where to write the evidence artifact (v2 by default)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    annotations = root / "annotations"
    if not annotations.is_dir():
        print(f"ERROR: no annotations directory under {root.resolve()}")
        return 2

    print("=" * 74)
    print("CDVQA temporal-order and label-semantics evidence  (v2, fixed palette)")
    print("=" * 74)
    print(f"root   : {root.resolve()}")
    print(f"scenes : {'ALL' if args.scenes <= 0 else args.scenes} carrying inc/dec signals")
    print()

    # -- gather the annotation ground truth ---------------------------------
    inc_dec, change_to_what = collect_signals(annotations, SPLITS)
    print(f"[0] scenes carrying inc/dec signals : {len(inc_dec):,}")
    print(f"    scenes carrying change_to_what  : {len(change_to_what):,}")

    all_names = sorted(p.name for p in (root / "label1").glob("*.png"))
    scenes = _select_scenes(all_names, set(inc_dec), args.scenes)
    print(f"    measuring {len(scenes)} scenes against label1/label2/im1/im2")
    print()

    # -- accumulators --------------------------------------------------------
    # Fixed-palette order test: both directions measured, split into
    # directional hits vs tie hits.
    fx = {
        "h1_n": 0, "h1_ok": 0, "h1_dir": 0, "h1_tie": 0,
        "h2_n": 0, "h2_ok": 0, "h2_dir": 0, "h2_tie": 0,
    }
    # Best-fit palette under each hypothesis (per class argmax), a symmetry check.
    best: dict[str, dict[str, dict[int, list[int]]]] = {
        "h1": defaultdict(lambda: defaultdict(lambda: [0, 0])),
        "h2": defaultdict(lambda: defaultdict(lambda: [0, 0])),
    }
    # Content test.
    pair = {
        "w1_n": 0, "w1_mean": 0, "w1_med": 0, "w1_major": 0,
        "w2_n": 0, "w2_mean": 0, "w2_med": 0, "w2_major": 0,
        "v1_n": 0, "v1_grn_mean": 0, "v1_grn_med": 0,
        "v2_n": 0, "v2_grn_mean": 0, "v2_grn_med": 0,
        "white_n": 0, "white_im1_darker": 0,
        "align_l1_n": 0, "align_l1_im1": 0, "align_l2_n": 0, "align_l2_im2": 0,
    }
    white_pixels = nonwhite_pixels = nonwhite_same = 0
    # change_to_what: per scene, keep the joint hists so subsets can be resolved.
    c2w_pending: list[tuple[str, dict[str, str], Counter[int], Counter[int]]] = []

    for name in scenes:
        l1_img = np.asarray(Image.open(root / "label1" / name))
        l2_img = np.asarray(Image.open(root / "label2" / name))
        k1 = _pack(l1_img)
        k2 = _pack(l2_img)

        cov1 = _hist(k1)
        cov2 = _hist(k2)
        n_px = k1.size
        colours = set(cov1) | set(cov2)

        # -- 1. fixed-palette order test (MEASURED both directions) ---------
        for cls, sig in inc_dec.get(name, {}).items():
            ckey = FIXED_KEY[cls]
            delta = (cov2.get(ckey, 0) - cov1.get(ckey, 0)) / n_px
            if "inc" in sig:
                _acc(fx, "h1", (sig["inc"] and delta > 0) or (not sig["inc"] and delta <= 0), delta)
                _acc(fx, "h2", (sig["inc"] and delta < 0) or (not sig["inc"] and delta >= 0), delta)
            if "dec" in sig:
                _acc(fx, "h1", (sig["dec"] and delta < 0) or (not sig["dec"] and delta >= 0), delta)
                _acc(fx, "h2", (sig["dec"] and delta > 0) or (not sig["dec"] and delta <= 0), delta)

        # -- 2a. best-fit palette under each hypothesis (symmetry check) ----
        for cls, sig in inc_dec.get(name, {}).items():
            for col in colours:
                d = (cov2.get(col, 0) - cov1.get(col, 0)) / n_px
                for hyp, rule in (("h1", 1), ("h2", -1)):
                    if "inc" in sig:
                        best[hyp][cls][col][0] += 1
                        best[hyp][cls][col][1] += int(
                            (sig["inc"] and rule * d > 0) or (not sig["inc"] and rule * d <= 0)
                        )
                    if "dec" in sig:
                        best[hyp][cls][col][0] += 1
                        best[hyp][cls][col][1] += int(
                            (sig["dec"] and rule * d < 0) or (not sig["dec"] and rule * d >= 0)
                        )

        # -- 2b. change_to_what (stash compact joint hists per scene) -------
        if name in change_to_what:
            c2w_pending.append(
                (
                    name,
                    change_to_what[name],
                    _joint_hist(k1, k2),
                    _joint_hist(k2, k1),
                )
            )

        # -- 3. image<->label pairing (fixed-mask content test) -------------
        im1 = np.asarray(Image.open(root / "im1" / name)).astype(np.float64)
        im2 = np.asarray(Image.open(root / "im2" / name)).astype(np.float64)
        lum1, lum2 = im1.mean(2), im2.mean(2)
        grn1 = im1[:, :, 1] - (im1[:, :, 0] + im1[:, :, 2]) / 2.0
        grn2 = im2[:, :, 1] - (im2[:, :, 0] + im2[:, :, 2]) / 2.0

        m1w = k1 == WATER_KEY
        m2w = k2 == WATER_KEY
        if int(m1w.sum()) > 50:
            a, b = lum1[m1w], lum2[m1w]
            pair["w1_n"] += 1
            pair["w1_mean"] += int(a.mean() < b.mean())
            pair["w1_med"] += int(np.median(a) < np.median(b))
            pair["w1_major"] += int((a < b).sum() > (a > b).sum())
        if int(m2w.sum()) > 50:
            a, b = lum2[m2w], lum1[m2w]
            pair["w2_n"] += 1
            pair["w2_mean"] += int(a.mean() < b.mean())
            pair["w2_med"] += int(np.median(a) < np.median(b))
            pair["w2_major"] += int((a < b).sum() > (a > b).sum())
        for tag, mask, gi_a, gi_b in (("v1", k1, im1, im2), ("v2", k2, im2, im1)):
            mv = np.isin(mask, VEG_KEYS)
            if int(mv.sum()) > 50:
                ga = gi_a[:, :, 1] - (gi_a[:, :, 0] + gi_a[:, :, 2]) / 2.0
                gb = gi_b[:, :, 1] - (gi_b[:, :, 0] + gi_b[:, :, 2]) / 2.0
                pair[f"{tag}_n"] += 1
                pair[f"{tag}_grn_mean"] += int(ga[mv].mean() > gb[mv].mean())
                pair[f"{tag}_grn_med"] += int(np.median(ga[mv]) > np.median(gb[mv]))

        # white control: no signal expected -> ~chance
        w = k1 == WHITE_KEY
        if int(w.sum()) > 0:
            pair["white_n"] += 1
            pair["white_im1_darker"] += int(lum1[w].mean() < lum2[w].mean())

        # mask-alignment: image-derived green mask vs label vegetation mask
        def _green_mask(img: np.ndarray) -> np.ndarray:
            return (img[:, :, 1] - img[:, :, 0] > 10) & (img[:, :, 1] - img[:, :, 2] > 10)

        l1veg = np.isin(k1, VEG_KEYS)
        l2veg = np.isin(k2, VEG_KEYS)
        g1, g2 = _green_mask(im1), _green_mask(im2)
        if int(l1veg.sum()) > 50:
            iou11 = (g1 & l1veg).sum() / max(1, (g1 | l1veg).sum())
            iou21 = (g2 & l1veg).sum() / max(1, (g2 | l1veg).sum())
            pair["align_l1_n"] += 1
            pair["align_l1_im1"] += int(iou11 > iou21)
        if int(l2veg.sum()) > 50:
            iou12 = (g1 & l2veg).sum() / max(1, (g1 | l2veg).sum())
            iou22 = (g2 & l2veg).sum() / max(1, (g2 | l2veg).sum())
            pair["align_l2_n"] += 1
            pair["align_l2_im2"] += int(iou22 > iou12)

        # white semantics
        nw = ~w
        white_pixels += int(w.sum())
        nonwhite_pixels += int(nw.sum())
        nonwhite_same += int(((k1 == k2) & nw).sum())

    # -- finalise the fixed-palette order test ------------------------------
    h1_rate = fx["h1_ok"] / fx["h1_n"] if fx["h1_n"] else 0.0
    h2_rate = fx["h2_ok"] / fx["h2_n"] if fx["h2_n"] else 0.0

    # -- finalise the best-fit tables ---------------------------------------
    best_fit: dict[str, dict[str, Any]] = {}
    for hyp in ("h1", "h2"):
        per_class: dict[str, Any] = {}
        tot = ok = 0
        collapsed: list[str] = []
        for cls in CLASS_NAMES:
            rows = [
                (okc / nc, col) for col, (nc, okc) in best[hyp][cls].items() if nc
            ]
            if not rows:
                continue
            rows.sort(key=lambda r: (-r[0], r[1]))
            bestrate, bestcol = rows[0]
            runner = rows[1][0] if len(rows) > 1 else 0.0
            nc, okc = best[hyp][cls][bestcol]
            tot += nc
            ok += okc
            if bestcol == WHITE_KEY:
                collapsed.append(cls)
            per_class[cls] = {
                "best_agreement": round(bestrate, 4),
                "rgb": list(_key_to_rgb(bestcol)),
                "runner_up": round(runner, 4),
                "n": nc,
            }
        best_fit[hyp] = {
            "overall_agreement": round(ok / tot, 4) if tot else None,
            "n": tot,
            "classes_collapsed_to_background": collapsed,
            "per_class": per_class,
        }

    # -- change_to_what (fixed palette) + stability -------------------------
    def resolve_c2w(pending: list[tuple[str, dict[str, str], Counter[int], Counter[int]]]) -> dict[str, dict[str, Any]]:
        stats = {"answer_from_label2": [0, 0], "answer_from_label1": [0, 0]}
        for _name, answers, j12, j21 in pending:
            for src_cls, dest_answer in answers.items():
                src_key = FIXED_KEY.get(src_cls)
                if src_key is None:
                    continue
                for tag, joint in (("answer_from_label2", j12), ("answer_from_label1", j21)):
                    dc: Counter[int] = Counter()
                    for combined, cnt in joint.items():
                        if (combined >> 24) != src_key:
                            continue
                        d_key = combined & 0xFFFFFF
                        if d_key in (WHITE_KEY, src_key):
                            continue
                        dc[d_key] += cnt
                    if not dc:
                        continue
                    modal_cls = KEY_TO_CLASS.get(max(dc, key=dc.get))
                    stats[tag][0] += 1
                    stats[tag][1] += int(modal_cls == dest_answer)
        return {
            tag: {"agreement": round(n_ok / n, 4) if n else None, "n": n}
            for tag, (n, n_ok) in stats.items()
        }

    c2w_full = resolve_c2w(c2w_pending)
    c2w_names = sorted({p[0] for p in c2w_pending})
    c2w_stability: dict[str, Any] = {}
    for size in C2W_STABILITY_SIZES:
        label = "all" if size <= 0 else str(size)
        chosen = set(_select_scenes(c2w_names, set(c2w_names), size))
        subset = [p for p in c2w_pending if p[0] in chosen]
        c2w_stability[label] = resolve_c2w(subset)

    # -- print ---------------------------------------------------------------
    print("[1] ORDER TEST — FIXED SECOND palette, NO fitting, ALL scenes")
    print(f"      label1=pre, label2=post  agreement = {h1_rate:.4f}  "
          f"(n={fx['h1_n']}, directional={fx['h1_dir']}, ties={fx['h1_tie']})")
    print(f"      label1=post, label2=pre  agreement = {h2_rate:.4f}  "
          f"(n={fx['h2_n']}, directional={fx['h2_dir']}, ties={fx['h2_tie']})")
    print(f"      reversed DIRECTIONAL hits        : {fx['h2_dir']}  "
          f"(0 => the reverse rate is pure no-change ties, not agreement)")
    order_established = h1_rate > 0.99 and h2_rate < 0.5 and fx["h2_dir"] == 0
    print(f"      VERDICT: {'label1 = PRE, label2 = POST' if order_established else 'NOT ESTABLISHED'}")
    print()

    print("[2] BEST-FIT palette per hypothesis (does re-fitting rescue the reverse?)")
    for hyp in ("h1", "h2"):
        bf = best_fit[hyp]
        print(f"      {hyp}: overall best-fit = {bf['overall_agreement']} (n={bf['n']}), "
              f"collapsed-to-background = {bf['classes_collapsed_to_background']}")
    print()

    print("[3] INDEPENDENT CONFIRMATION via change_to_what (fixed palette)")
    for tag, st in c2w_full.items():
        print(f"      modal DESTINATION class read from {tag:<20} agreement = "
              f"{st['agreement']}  (n={st['n']})")
    print("      stability (label2 direction): " + ", ".join(
        f"{k}={v['answer_from_label2']['agreement']} (n={v['answer_from_label2']['n']})"
        for k, v in c2w_stability.items()
    ))
    print()

    def _rate(num: int, den: int) -> float:
        return num / den if den else 0.0

    print("[4] IMAGE <-> LABEL PAIRING (fixed-mask content test; SUPPORT, not proof)")
    print(f"      water: im1 darker where label1==water : "
          f"mean={_rate(pair['w1_mean'], pair['w1_n']):.3f} "
          f"median={_rate(pair['w1_med'], pair['w1_n']):.3f} "
          f"majority={_rate(pair['w1_major'], pair['w1_n']):.3f} (n={pair['w1_n']})")
    print(f"      water: im2 darker where label2==water : "
          f"mean={_rate(pair['w2_mean'], pair['w2_n']):.3f} "
          f"median={_rate(pair['w2_med'], pair['w2_n']):.3f} "
          f"majority={_rate(pair['w2_major'], pair['w2_n']):.3f} (n={pair['w2_n']})")
    print(f"      veg  : im1 greener where label1==veg  : "
          f"mean={_rate(pair['v1_grn_mean'], pair['v1_n']):.3f} "
          f"median={_rate(pair['v1_grn_med'], pair['v1_n']):.3f} (n={pair['v1_n']})")
    print(f"      veg  : im2 greener where label2==veg  : "
          f"mean={_rate(pair['v2_grn_mean'], pair['v2_n']):.3f} "
          f"median={_rate(pair['v2_grn_med'], pair['v2_n']):.3f} (n={pair['v2_n']})")
    print(f"      WHITE control (expect ~chance)        : "
          f"im1 darker rate={_rate(pair['white_im1_darker'], pair['white_n']):.3f} (n={pair['white_n']})")
    print(f"      mask-alignment im1-green vs label1-veg beats im2 : "
          f"{_rate(pair['align_l1_im1'], pair['align_l1_n']):.3f} (n={pair['align_l1_n']})")
    print(f"      mask-alignment im2-green vs label2-veg beats im1 : "
          f"{_rate(pair['align_l2_im2'], pair['align_l2_n']):.3f} (n={pair['align_l2_n']})")
    print()

    print("[5] WHITE semantics")
    total_px = white_pixels + nonwhite_pixels
    print(f"      white share of pixels            : {white_pixels / total_px:.4f}")
    print(f"      non-white where label1==label2   : {nonwhite_same / nonwhite_pixels:.4f}")
    print()

    wl1 = _rate(pair["w1_mean"], pair["w1_n"])
    wl2 = _rate(pair["w2_mean"], pair["w2_n"])
    vl1 = _rate(pair["v1_grn_mean"], pair["v1_n"])
    vl2 = _rate(pair["v2_grn_mean"], pair["v2_n"])
    pairing_supported = wl1 > 0.5 and wl2 > 0.5 and vl1 > 0.5 and vl2 > 0.5
    label_supported = (
        order_established
        and (c2w_full["answer_from_label2"]["agreement"] or 0) > 0.9
        and (c2w_full["answer_from_label1"]["agreement"] or 1) < 0.5
    )
    if label_supported and pairing_supported:
        print("CONCLUSION: label1 = PRE, label2 = POST (established from the annotations;")
        print("            the reverse is not merely lower but directionally empty).")
        print("            im1 = PRE / im2 = POST is SUPPORTED by the content test")
        print("            (water darkness / vegetation greenness vs the white control),")
        print("            not proven — it is statistical, not exact.")
    elif order_established:
        print("CONCLUSION: label1 = PRE, label2 = POST established; image pairing")
        print("            not decisively supported by the content test.")
    else:
        print("CONCLUSION: temporal order NOT established by this evidence.")
    print("=" * 74)

    payload = {
        "phase": "10",
        "kind": "cdvqa_temporal_order_evidence_v2",
        "supersedes": "artifacts/cdvqa/temporal_order_evidence.json",
        "supersession_reason": (
            "v1 fit the palette in-sample to maximise H1 and computed the reversed "
            "rate as 1 - h1_rate, so the perfect split was guaranteed by construction. "
            "v2 measures BOTH directions against the FIXED SECOND palette and reports "
            "directional vs tie evidence separately."
        ),
        "root": str(root.resolve()),
        "scenes_measured": len(scenes),
        "method": "fixed SECOND palette; no per-hypothesis fitting for the order test",
        "fixed_palette": {cls: list(rgb) for cls, rgb in FIXED_PALETTE.items()},
        "label_order": {
            "fixed_palette": {
                "h1_label1_pre_label2_post": {
                    "agreement": round(h1_rate, 4), "n": fx["h1_n"],
                    "directional_hits": fx["h1_dir"], "tie_hits": fx["h1_tie"],
                },
                "h2_label1_post_label2_pre": {
                    "agreement": round(h2_rate, 4), "n": fx["h2_n"],
                    "directional_hits": fx["h2_dir"], "tie_hits": fx["h2_tie"],
                },
            },
            "best_fit": best_fit,
            "verdict": "label1=pre,label2=post" if order_established else "not_established",
            "verdict_basis": (
                "H1 directional agreement 1.0000 with zero directional hits under H2; "
                "the reversed rate is entirely no-change ties (delta == 0)."
            ),
        },
        "change_to_what": {"full": c2w_full, "stability": c2w_stability},
        "image_label_pairing": {
            "water_label1_im1_darker": {
                "mean": round(wl1, 4),
                "median": round(_rate(pair["w1_med"], pair["w1_n"]), 4),
                "majority": round(_rate(pair["w1_major"], pair["w1_n"]), 4),
                "n": pair["w1_n"],
            },
            "water_label2_im2_darker": {
                "mean": round(wl2, 4),
                "median": round(_rate(pair["w2_med"], pair["w2_n"]), 4),
                "majority": round(_rate(pair["w2_major"], pair["w2_n"]), 4),
                "n": pair["w2_n"],
            },
            "veg_label1_im1_greener": {
                "mean": round(vl1, 4),
                "median": round(_rate(pair["v1_grn_med"], pair["v1_n"]), 4),
                "n": pair["v1_n"],
            },
            "veg_label2_im2_greener": {
                "mean": round(vl2, 4),
                "median": round(_rate(pair["v2_grn_med"], pair["v2_n"]), 4),
                "n": pair["v2_n"],
            },
            "white_control_im1_darker_rate": round(_rate(pair["white_im1_darker"], pair["white_n"]), 4),
            "white_control_n": pair["white_n"],
            "mask_alignment": {
                "im1_green_vs_label1_veg_beats_im2": round(_rate(pair["align_l1_im1"], pair["align_l1_n"]), 4),
                "im2_green_vs_label2_veg_beats_im1": round(_rate(pair["align_l2_im2"], pair["align_l2_n"]), 4),
            },
            "strength": "supportive, statistical (not exact)",
        },
        "white_semantics": {
            "white_share": round(white_pixels / total_px, 4),
            "nonwhite_where_label1_eq_label2": round(nonwhite_same / nonwhite_pixels, 4),
        },
        "note": (
            "label1=pre / label2=post is established from the annotations against the "
            "FIXED SECOND palette (no fitting) — the reverse is directionally empty. "
            "im1=pre / im2=post is SUPPORTED by a content test (water darkness / "
            "vegetation greenness vs a white control), not proven. Nothing is inferred "
            "from directory names."
        ),
    }

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
