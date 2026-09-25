"""VRSBench benchmark adapter — referring-expression visual grounding.

WHAT THE CORPUS ACTUALLY IS (measured 2026-09-23)
-------------------------------------------------
`training/data/vrsbench/` holds the real VRSBench download:

    VRSBench_EVAL_referring.json   16,159 referring records (the held-out set)
    VRSBench_train.json            142,390 interleaved conversations
    images/Images_val/             9,350 extracted PNG  <- the eval imagery
    Images_val.zip                 3.97 GB archive (the same imagery, zipped)
    Images_train.zip, Images_train/

Measured match rate: **16,159 of 16,159** eval records have their image present
in `images/Images_val/`. The eval split is therefore fully evaluable, which is
why this adapter can be exercised end to end rather than only declared.

THE BOX CONVENTION IS THE WHOLE RISK
------------------------------------
VRSBench records the ground-truth box as a **0-100 xyxy token string**:

    "ground_truth": "{<25><40><33><60>}"   ->   [0.25, 0.40, 0.33, 0.60]

Treating those numbers as pixels or as 0-1 produces boxes a hundred times too
large, and the failure mode is a *suspiciously high IoU*, not a crash. The
conversion is therefore never inline: `training.data.vrsbench` owns the parse and
`evaluation.metrics.grounding.benchmark_to_normalized` owns the scale, and this
adapter calls both rather than re-deriving either.

WHY THE LOADER IS IMPORTED, NOT REIMPLEMENTED
---------------------------------------------
`training/data/vrsbench.py` already handles three schemas (verified-eval,
verified-train, legacy key-fallback), the noisy train boxes, and the O(n^2) walk
trap. A second parser here would be a second definition of the VRSBench format.
This module adds only the adapter contract around it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evaluation.benchmark_adapters._common import (
    AdapterDataError,
    as_split_name,
    check_identity_leakage,
    provenance_block,
    resolve_corpus_root,
    sample_manifest,
    subset_policy_block,
)
from evaluation.benchmark_adapters.base import (
    BenchmarkAdapter,
    BenchmarkNotAvailableError,
    BenchmarkSample,
    BenchmarkSource,
    BenchmarkStatus,
    CorpusDescription,
)
from evaluation.metrics import grounding as grounding_metrics
from core.code_revision import REPO_ROOT

__all__ = ["VrsBenchAdapter", "make_scorer", "EVAL_JSON_NAME", "is_degenerate_box"]

#: The eval annotation file, named explicitly. `training.data.vrsbench`'s default
#: scan is shallowest-first, which alphabetically selects the EVAL file before
#: the TRAIN file -- a leakage trap its own docstring calls out. Naming the file
#: is the explicit act that avoids it.
EVAL_JSON_NAME = "VRSBench_EVAL_referring.json"

#: VRSBench's published eval split size, for the coverage record.
OFFICIAL_EVAL_RECORDS = 16159

_ROOT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("data/VRSBench", "canonical target (declared.py)"),
    ("training/data/vrsbench", "discovered corpus"),
)

#: IoU at or above which a prediction counts as a hit for `accuracy`.
HIT_IOU = 0.5


def is_degenerate_box(box: Any) -> bool:
    """Whether a box has zero width or zero height, so it has no area.

    Added 2026-09-23 after measuring the corpus. 13 of the 16,159 EVAL records
    carry a degenerate gold box -- e.g. `[0.92, 0.0, 0.96, 0.0]` (y1 == y2) --
    produced by the 0-100 token convention where two of the four tokens are
    equal. Such a box cannot be hit: IoU against it is 0/0, and every convention
    treats that as 0. A perfect predictor therefore scores 16,146/16,159 =
    0.99920 rather than 1.0, and without this being stated the shortfall looks
    like a prediction error rather than a property of the gold data.

    This is a DATA observation, not a parsing bug: the adapter reproduces the
    corpus faithfully. `degenerate_gold()` reports the affected sample ids so the
    shortfall is attributable.
    """
    if box is None or len(list(box)) != 4:
        return False
    x1, y1, x2, y2 = (float(v) for v in box)
    return x2 <= x1 or y2 <= y1


class VrsBenchAdapter(BenchmarkAdapter):
    """Referring-expression grounding on VRSBench's held-out eval split."""

    name = "vrsbench"
    ADAPTER_VERSION = "1.0.0"

    source = BenchmarkSource(
        name="vrsbench",
        root=REPO_ROOT / "data" / "VRSBench",
        citation="VRSBench (lx709), as recorded in the implementation plan",
        note=(
            "Canonical target root. The corpus is present at "
            "`training/data/vrsbench/` rather than here."
        ),
    )

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        hit_iou: float = HIT_IOU,
        limit: int | None = None,
    ) -> None:
        self._explicit_root = Path(root) if root is not None else None
        self.hit_iou = float(hit_iou)
        self.limit = limit
        self._resolved: Path | None = None
        self._locations: list = []
        self._resolved_once = False

    # -- corpus ------------------------------------------------------------
    def _resolve(self):
        if self._explicit_root is not None:
            path = self._explicit_root
            return (path if path.exists() else None), []
        if not self._resolved_once:
            self._resolved, self._locations = resolve_corpus_root(_ROOT_CANDIDATES)
            self._resolved_once = True
        return self._resolved, self._locations

    def describe_corpus(self) -> CorpusDescription:
        root, locations = self._resolve()
        if root is None:
            return CorpusDescription(
                benchmark=self.name,
                available=False,
                root=str(_ROOT_CANDIDATES[0][0]),
                reason=(
                    "no VRSBench root found. Searched: "
                    + ", ".join(str(loc.path) for loc in locations)
                ),
                note="A VRSBench download holds VRSBench_EVAL_referring.json and the eval images.",
            )

        annotation = root / EVAL_JSON_NAME
        image_dirs = [root / "images" / "Images_val", root / "Images_val", root / "images"]
        n_images = 0
        image_root: Path | None = None
        for candidate in image_dirs:
            if candidate.is_dir():
                n_images = len(list(candidate.glob("*.png"))) + len(
                    list(candidate.glob("*.jpg"))
                )
                if n_images:
                    image_root = candidate
                    break

        available = annotation.is_file()
        return CorpusDescription(
            benchmark=self.name,
            available=available,
            root=str(root),
            n_files=n_images,
            bytes=0,
            kinds={"annotation": EVAL_JSON_NAME, "images": str(image_root or "")},
            reason=(
                None
                if available
                else f"{EVAL_JSON_NAME} not found under {root}"
            ),
            note=(
                f"resolved root: {root}; annotation present: {available}; "
                f"eval images present: {n_images} at {image_root}"
            ),
        )

    def is_official(self) -> bool:
        """True only when the corpus answered at the canonical declared root."""
        root, _ = self._resolve()
        return root is not None and root == (REPO_ROOT / "data" / "VRSBench")

    def corpus_kind(self) -> BenchmarkStatus:
        root, _ = self._resolve()
        if root is None:
            return BenchmarkStatus.NOT_RUN
        return BenchmarkStatus.REAL if self.is_official() else BenchmarkStatus.DEGRADED

    def subset_policy(self) -> dict[str, Any]:
        """Whether the eval split present here is the full published split.

        Two independent things are recorded, because they fail separately:
        whether the record COUNT is complete, and whether the corpus sits at the
        canonical root. A complete split at a non-canonical path is DEGRADED for
        location, not for coverage -- saying "incomplete" there would be wrong.
        """
        root, _ = self._resolve()
        n_present = None
        if root is not None:
            annotation = root / EVAL_JSON_NAME
            if annotation.is_file():
                import json

                try:
                    n_present = len(json.loads(annotation.read_text(encoding="utf-8")))
                except Exception:  # noqa: BLE001
                    n_present = None

        count_complete = n_present == OFFICIAL_EVAL_RECORDS
        at_canonical_root = self.is_official()

        if count_complete and at_canonical_root:
            basis = (
                f"record count in {EVAL_JSON_NAME} equals the published eval split "
                f"size ({OFFICIAL_EVAL_RECORDS}) AND the corpus is at the canonical "
                f"declared root"
            )
        elif count_complete and not at_canonical_root:
            basis = (
                f"record count in {EVAL_JSON_NAME} equals the published eval split "
                f"size ({OFFICIAL_EVAL_RECORDS}), but the corpus is NOT at the "
                f"canonical declared root ({self.source.root}); DEGRADED for "
                f"location, not for coverage"
            )
        else:
            basis = (
                f"record count in {EVAL_JSON_NAME} is {n_present}, which does not "
                f"equal the published eval split size ({OFFICIAL_EVAL_RECORDS})"
            )

        return subset_policy_block(
            is_full_corpus=count_complete and at_canonical_root,
            n_present=n_present,
            n_official=OFFICIAL_EVAL_RECORDS,
            basis=basis,
            detail=(
                "Only the referring-expression eval split is scored here. The "
                "caption/VQA tasks in VRSBench_train.json belong to other "
                "components and are not grounding samples."
            ),
        )

    # -- loading -----------------------------------------------------------
    def load(self, *, split: str = "test") -> list[BenchmarkSample]:
        """Package the eval split's referring expressions as adapter samples.

        Images are NOT opened. The verified loader's `open_images=False` mode is
        used because the full 16,159-record split costs ~12.7 GB if every PIL
        object is held at once -- a figure the loader's own docstring records
        from measurement. Samples carry `image_path` and the caller opens on
        demand.
        """
        if not self.describe_corpus().available:
            description = self.describe_corpus()
            raise BenchmarkNotAvailableError(
                f"benchmark {self.name!r} is not available: "
                f"{description.reason or 'corpus not present'} "
                f"(root={description.root})",
                context=description.to_dict(),
            )

        root, _ = self._resolve()
        assert root is not None

        from training.data.vrsbench import load_referring_samples

        referring = load_referring_samples(
            root,
            limit=self.limit,
            verbose=False,
            open_images=False,
            json_name=EVAL_JSON_NAME,
        )

        canonical = as_split_name(split)
        samples: list[BenchmarkSample] = []
        for sample in referring:
            samples.append(
                BenchmarkSample(
                    sample_id=str(sample.sample_id),
                    payload={
                        "image_path": str(sample.image_path) if sample.image_path else None,
                        "phrase": sample.phrase,
                    },
                    # The ground-truth box, already in SatQuery's 0-1 convention.
                    expected=list(sample.box),
                    meta={
                        "split": canonical,
                        "split_label": split,
                        "scene": Path(str(sample.image_path)).stem
                        if sample.image_path
                        else str(sample.sample_id),
                        "dataset": self.name,
                        "source_file": sample.source_file,
                    },
                )
            )
        return samples

    # -- task shape --------------------------------------------------------
    def metric_names(self) -> tuple[str, ...]:
        """`accuracy` and `iou`, both registered in `evaluation.normalize`.

        Recall at other IoU thresholds is deliberately NOT emitted as a
        normalised metric: the project's `normalize.TRANSFORMS` registry has no
        `recall@0.75` entry, and adding one would be a contract decision rather
        than an adapter's choice. The full grounding report -- which carries
        Recall@0.25/0.50/0.75 -- is available from `grounding_report()`.
        """
        return ("accuracy", "iou")

    def metric_definitions(self) -> dict[str, str]:
        return {
            "accuracy": f"fraction of samples with best IoU >= {self.hit_iou}",
            "iou": "mean best IoU across samples",
            "box_convention": "0-1 xyxy after the documented 0-100 conversion",
            "hit_iou": str(self.hit_iou),
        }

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        samples: Sequence[BenchmarkSample],
        *,
        predict: Callable[[BenchmarkSample], Any],
    ) -> dict[str, float]:
        """Score referring predictions against the ground-truth boxes.

        Args:
            samples: samples from `load()`.
            predict: maps a sample to a single `[x1, y1, x2, y2]` box in the 0-1
                convention. **Required** -- a default would have to invent boxes.

        Raises:
            AdapterDataError: no samples, or a prediction that is not a 4-value
                box. A malformed box is rejected rather than repaired; repairing
                it would silently move the box.

        DEGENERATE GOLD BOXES (documented 2026-09-23)
        --------------------------------------------
        A gold box with zero width or height cannot be hit -- IoU against it is
        0/0, which is scored as 0 here, matching the convention used everywhere
        else. 13 of the corpus's 16,159 EVAL records are in that state, so the
        accuracy ceiling is 16,146/16,159 = 0.99920, not 1.0. The convention is
        kept (excluding them would change the metric's denominator) and the
        affected ids are reported separately by `degenerate_gold()`, so the
        shortfall is attributable to the data rather than read as a model error.
        """
        if not samples:
            raise AdapterDataError(
                "no samples to score; returning 0.0 here would look like a score"
            )

        hits = 0
        ious: list[float] = []
        for sample in samples:
            truth = list(sample.expected or [])
            if len(truth) != 4:
                raise AdapterDataError(
                    f"sample {sample.sample_id!r}: ground-truth box is not 4 values: "
                    f"{truth}"
                )
            prediction = predict(sample)
            box = list(prediction)
            if len(box) != 4:
                raise AdapterDataError(
                    f"sample {sample.sample_id!r}: prediction must be 4 values "
                    f"[x1,y1,x2,y2], got {box}"
                )
            iou = float(grounding_metrics.box_iou(box, truth))
            ious.append(iou)
            if iou >= self.hit_iou:
                hits += 1

        return {
            "accuracy": hits / len(samples),
            "iou": sum(ious) / len(ious),
        }

    def degenerate_gold(self, samples: Sequence[BenchmarkSample]) -> dict[str, Any]:
        """Report gold boxes that have no area and therefore cannot be hit.

        Kept out of `evaluate()`'s return value on purpose: that dict's keys are
        `metric_names()`, and adding a non-metric key would be recorded as a
        normalisation failure for every run. This is corpus diagnostics, so it
        travels in the manifest and the subset policy instead.
        """
        ids = [
            s.sample_id for s in samples if is_degenerate_box(s.expected)
        ]
        return {
            "n_samples": len(samples),
            "n_degenerate_gold": len(ids),
            "sample_ids": ids,
            "accuracy_ceiling": (
                (len(samples) - len(ids)) / len(samples) if samples else None
            ),
            "policy": (
                "a zero-area gold box scores IoU 0 by the standard 0/0 convention, "
                "so it is counted as a miss rather than dropped; dropping it would "
                "change the metric's denominator without saying so"
            ),
        }

    def grounding_report(
        self,
        samples: Sequence[BenchmarkSample],
        *,
        predict: Callable[[BenchmarkSample], Any],
        thresholds: Sequence[float] = grounding_metrics.DEFAULT_THRESHOLDS,
    ):
        """The project's full grounding report, for Recall@IoU inspection.

        Separate from `evaluate()` because the report carries thresholds that are
        not in the normalisation registry. Exposed so a caller can see
        Recall@0.25/0.50/0.75 without those names leaking into a normalised
        metric dict.
        """
        per_image = []
        for sample in samples:
            per_image.append(([list(predict(sample))], [list(sample.expected or [])]))
        return grounding_metrics.score_dataset(per_image, thresholds=thresholds)

    # -- reporting ---------------------------------------------------------
    def manifest(self, *, split: str = "test") -> dict[str, Any]:
        samples = self.load(split=split)
        root, _ = self._resolve()
        return sample_manifest(
            dataset=self.name,
            version=self._version_string(root),
            split=as_split_name(split),
            sample_ids=[s.sample_id for s in samples],
            root=root,
            extra={
                "split_label": split,
                "annotation_file": EVAL_JSON_NAME,
                "subset_policy": self.subset_policy(),
                "n_images_referenced": len(
                    {s.payload.get("image_path") for s in samples}
                ),
                "gold_quality": self.degenerate_gold(samples),
            },
        )

    def _version_string(self, root: Path | None) -> str:
        if root is None:
            return "unavailable (no corpus root)"
        return (
            "VRSBench (schema VERIFIED 2026-09-16 against the official mirror "
            "xiang709/VRSBench); revision not pinned in-repo"
        )

    def provenance(
        self, *, split: str = "test", config_hash: str | None = None
    ) -> dict[str, Any]:
        root, _ = self._resolve()
        return provenance_block(
            adapter=type(self).__name__,
            adapter_version=self.ADAPTER_VERSION,
            dataset=self.name,
            version=self._version_string(root),
            split=as_split_name(split),
            manifest=self.manifest(split=split),
            config_hash=config_hash,
            metric_definitions=self.metric_definitions(),
            extra={
                "resolved_root": str(root) if root else None,
                "is_official": self.is_official(),
                "hit_iou": self.hit_iou,
                "box_scale": 100.0,
            },
        )

    def leakage_check(self) -> dict[str, Any]:
        """Assert the eval records share no image with the train file.

        VRSBench ships train and eval as separate JSON files over separate image
        archives, so this is a real check rather than a formality: a mis-zipped
        download could put eval imagery in the train archive, and the adapter
        would then report a training-contaminated number.
        """
        root, _ = self._resolve()
        if root is None:
            return {
                "label": f"{self.name}:eval-vs-train",
                "clean": None,
                "performed": False,
                "note": "corpus not resolved; leakage check not performed",
            }

        import json

        eval_path = root / EVAL_JSON_NAME
        train_path = root / "VRSBench_train.json"
        if not eval_path.is_file():
            return {
                "label": f"{self.name}:eval-vs-train",
                "clean": None,
                "performed": False,
                "note": f"{EVAL_JSON_NAME} absent; check not performed",
            }

        eval_ids = {
            Path(str(r.get("image_id", ""))).stem.lower()
            for r in json.loads(eval_path.read_text(encoding="utf-8"))
        }
        train_ids: set[str] = set()
        if train_path.is_file():
            train_ids = {
                Path(str(r.get("image", ""))).stem.lower()
                for r in json.loads(train_path.read_text(encoding="utf-8"))
            }

        report = check_identity_leakage(
            train_ids, eval_ids, label=f"{self.name}:train-vs-eval"
        )
        report["performed"] = True
        report["train_file_present"] = train_path.is_file()
        return report

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["adapter_version"] = self.ADAPTER_VERSION
        payload["corpus_kind"] = self.corpus_kind().value
        payload["subset_policy"] = (
            self.subset_policy() if self.describe_corpus().available else None
        )
        return payload


def make_scorer(
    adapter: VrsBenchAdapter,
    predict: Callable[[BenchmarkSample], Any],
) -> Callable[[Sequence[BenchmarkSample]], Mapping[str, Any]]:
    """Adapt `adapter.evaluate` into the runner's `Scorer` signature."""

    def scorer(samples: Sequence[BenchmarkSample]) -> Mapping[str, Any]:
        return adapter.evaluate(samples, predict=predict)

    return scorer
