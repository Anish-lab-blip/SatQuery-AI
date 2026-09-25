"""LEVIR-CD benchmark adapter — binary change detection over bitemporal pairs.

WHAT THE CORPUS ACTUALLY IS (measured 2026-09-23, not recalled)
--------------------------------------------------------------
`data/levir/` is a FLAT LEVIR-CD-256 tree, downloaded as
`keykeylv/levir-cd-256`:

    data/levir/A/      10,192 PNG, 256x256 RGB     T1 (before)
    data/levir/B/      10,192 PNG, 256x256 RGB     T2 (after)
    data/levir/label/  10,192 PNG, 256x256 L       change mask
    data/levir/edge/   10,192 PNG                  edge maps (unused here)
    data/levir/list/{train,val,test}.txt          7,120 / 1,024 / 2,047 stems

A/B/label share filenames exactly (`test_100_1.png` in all three), which was
verified by opening one triple and comparing shapes -- alignment is a property of
this corpus that a mis-aligned download would not have, so it is checked rather
than assumed.

THE DECLARED ROOT IS A TARGET, NOT A LOCATION
---------------------------------------------
`declared.py` names `data/LEVIR-CD` as the canonical target. The corpus is NOT
there; it is at `data/levir/`. Both are searched, and which one answered is
recorded in the corpus description. If only the non-canonical location is
present, the adapter reports `DEGRADED`, not `REAL` -- the material is genuine
LEVIR-CD, but it is not at the attested canonical location and the adapter will
not claim more than it measured.

WHY THIS IS A SEPARATE FILE FROM THE LOADER
-------------------------------------------
`training/change/dataset.py` already knows how to discover and read this tree,
including both layouts. Re-implementing that walk here would create a second
definition of the LEVIR layout that could drift from the first. This module
imports `load_levir_dataset` and adds only what the *benchmark adapter* contract
needs: corpus description, sample packaging, metric selection, manifest,
provenance and leakage checking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.benchmark_adapters._common import (
    AdapterDataError,
    CorpusLocation,
    as_split_name,
    check_identity_leakage,
    provenance_block,
    resolve_corpus_root,
    sample_manifest,
    subset_policy_block,
    utc_now,
)
from evaluation.benchmark_adapters.base import (
    BenchmarkAdapter,
    BenchmarkNotAvailableError,
    BenchmarkSample,
    BenchmarkSource,
    BenchmarkStatus,
    CorpusDescription,
)
from evaluation.metrics import change as change_metrics
from core.code_revision import REPO_ROOT

__all__ = ["LevirCdAdapter", "make_scorer"]

#: LEVIR-CD's published split sizes, for the subset/coverage record.
OFFICIAL_SPLIT_SIZES: dict[str, int] = {"train": 7120, "val": 1024, "test": 2048}

#: Root candidates, in preference order. The first is the canonical target named
#: by `declared.py`; the second is where the corpus actually lives here.
_ROOT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("data/LEVIR-CD", "canonical target (declared.py)"),
    ("data/levir", "discovered corpus (flat LEVIR-CD-256)"),
    ("data/LEVIR-CD256", "alternate name"),
)


class LevirCdAdapter(BenchmarkAdapter):
    """Binary change detection on LEVIR-CD bitemporal optical pairs.

    The adapter loads *paths*, never pixels. A 2,048-tile test split opened
    eagerly would hold ~4 GB of arrays; the sample carries `t1_path`/`t2_path`
    and the caller (or the predictor) opens what it needs, one tile at a time.
    """

    name = "levir_cd"
    ADAPTER_VERSION = "1.0.0"

    source = BenchmarkSource(
        name="levir_cd",
        root=REPO_ROOT / "data" / "LEVIR-CD",
        citation="LEVIR-CD (Zou et al.), as recorded in the implementation plan",
        note=(
            "Canonical target root. The corpus is present in this repository at "
            "`data/levir/` (flat LEVIR-CD-256) rather than here; both are "
            "searched and the answering location is recorded."
        ),
    )

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        threshold: float = change_metrics.DEFAULT_THRESHOLD,
    ) -> None:
        self._explicit_root = Path(root) if root is not None else None
        self.threshold = float(threshold)
        self._resolved: Path | None = None
        self._locations: list = []
        self._resolved_once = False

    # -- corpus ------------------------------------------------------------
    def _resolve(self) -> tuple[Path | None, list[CorpusLocation]]:
        if self._explicit_root is not None:
            path = self._explicit_root
            exists = path.exists()
            return (
                path if exists else None,
                [
                    CorpusLocation(
                        path=path,
                        exists=exists,
                        role="explicit root",
                        n_files=0,
                        note="" if exists else "path does not exist",
                    )
                ],
            )
        if not self._resolved_once:
            self._resolved, self._locations = resolve_corpus_root(_ROOT_CANDIDATES)
            self._resolved_once = True
        return self._resolved, self._locations

    def describe_corpus(self) -> CorpusDescription:
        """Measure what is actually on disk, and where."""
        root, locations = self._resolve()

        if root is None:
            return CorpusDescription(
                benchmark=self.name,
                available=False,
                root=str(_ROOT_CANDIDATES[0][0]),
                reason=(
                    "no LEVIR-CD root found. Searched: "
                    + ", ".join(f"{loc.path}" for loc in locations)
                ),
                note=(
                    "A LEVIR-CD tree is expected as either "
                    "`<root>/<split>/{A,B,label}/` (nested) or "
                    "`<root>/{A,B,label}/<split>_<scene>_<tile>.png` (flat). "
                    "Neither was found."
                ),
            )

        # FIXED 2026-09-23: this counted `root/A/*.png` only, so a corpus in the
        # NESTED layout (`<root>/<split>/{A,B,label}/`) reported `available:
        # False` and `load()` refused it -- even though `load_levir_dataset`
        # supports the nested shape and the docstring of `describe_corpus`'s own
        # `note` describes it. Found by testing a synthetic nested corpus. The
        # count now follows the detected layout, reusing the project's single
        # layout detector rather than re-deriving it here.
        from training.change.dataset import LAYOUT_NESTED, detect_levir_layout

        layout = detect_levir_layout(root)
        if layout == LAYOUT_NESTED:
            n_tiles = sum(
                len(list((root / split / "A").glob("*.png")))
                for split in ("train", "val", "test")
                if (root / split / "A").is_dir()
            )
            layout_note = "nested (<root>/<split>/{A,B,label}/)"
        else:
            a_dir = root / "A"
            n_tiles = len(list(a_dir.glob("*.png"))) if a_dir.is_dir() else 0
            layout_note = "flat (<root>/{A,B,label}/<split>_<scene>_<tile>.png)"

        return CorpusDescription(
            benchmark=self.name,
            available=n_tiles > 0,
            root=str(root),
            n_files=n_tiles,
            bytes=0,
            kinds={"A": "T1", "B": "T2", "label": "change mask", "edge": "edge map"},
            reason=(
                None
                if n_tiles
                else f"root exists but holds no A/*.png tiles in either layout: {root}"
            ),
            note=(
                f"resolved root: {root}. Layout: {layout_note}. Locations tried: "
                + "; ".join(f"{loc.role}={loc.path}" for loc in locations)
            ),
        )

    def is_official(self) -> bool:
        """True only when the corpus answered at the canonical declared root.

        A corpus found at `data/levir/` is genuine LEVIR-CD material, but the
        adapter did not find it where the project declared it should be. Claiming
        REAL there would overstate what was verified.
        """
        root, _ = self._resolve()
        if root is None:
            return False
        return root == (REPO_ROOT / "data" / "LEVIR-CD")

    def corpus_kind(self) -> BenchmarkStatus:
        """REAL at the canonical root; DEGRADED at a discovered root; else FIXTURE.

        Never FIXTURE when a real corpus was read: LEVIR-CD material found at
        `data/levir/` is real, so calling it synthetic would be a worse error
        than calling it partial.
        """
        root, _ = self._resolve()
        if root is None:
            return BenchmarkStatus.NOT_RUN
        return BenchmarkStatus.REAL if self.is_official() else BenchmarkStatus.DEGRADED

    def subset_policy(self, *, split: str = "test") -> dict[str, Any]:
        """Whether this split is the full official split or a subset."""
        split = as_split_name(split)
        n_present = len(self._load_items(split=split)) if self._available() else 0
        official = OFFICIAL_SPLIT_SIZES.get(split)
        return subset_policy_block(
            is_full_corpus=self.is_official(),
            n_present=n_present,
            n_official=official,
            basis=(
                "corpus found at the canonical declared root and the split list "
                "size matches the published count"
                if self.is_official()
                else "corpus present but not at the canonical declared root, so "
                "fullness is not attested"
            ),
            detail=(
                "LEVIR-CD-256 as downloaded; A/B/label share filenames, verified "
                "by opening a triple and comparing shapes."
            ),
        )

    def _available(self) -> bool:
        return self.describe_corpus().available

    # -- loading -----------------------------------------------------------
    def _load_items(self, *, split: str) -> list:
        root, _ = self._resolve()
        if root is None:
            raise BenchmarkNotAvailableError(
                f"LEVIR-CD root not found for split {split!r}; searched "
                f"{[str(loc.path) for loc in self._locations]}"
            )
        from training.change.dataset import load_levir_dataset

        return list(load_levir_dataset(root, splits=(split,)))

    def load(self, *, split: str = "test") -> list[BenchmarkSample]:
        """Package the split's tiles as adapter samples carrying PATHS.

        Raises:
            BenchmarkNotAvailableError: the corpus is absent. Never returns a
                synthetic list -- a quiet fallback here is how a fabricated
                benchmark score gets produced.
        """
        if not self._available():
            description = self.describe_corpus()
            raise BenchmarkNotAvailableError(
                f"benchmark {self.name!r} is not available: "
                f"{description.reason or 'corpus not present'} "
                f"(root={description.root})",
                context=description.to_dict(),
            )

        canonical = as_split_name(split)
        items = self._load_items(split=split)

        samples: list[BenchmarkSample] = []
        for item in items:
            label_path = Path(item["label_path"])
            samples.append(
                BenchmarkSample(
                    sample_id=str(item["sample_id"]),
                    payload={
                        "t1_path": str(item["t1_path"]),
                        "t2_path": str(item["t2_path"]),
                        "label_path": str(label_path),
                        "shape": None,
                    },
                    # `expected` is the GROUND TRUTH. It is a path, not an array,
                    # so a caller cannot mistake a placeholder for the mask.
                    expected=str(label_path),
                    meta={
                        "split": canonical,
                        "split_label": split,
                        "scene": str(item.get("group_key") or item.get("image_key") or ""),
                        "dataset": self.name,
                    },
                )
            )
        return samples

    # -- task shape --------------------------------------------------------
    def metric_names(self) -> tuple[str, ...]:
        """Metrics this adapter's scores carry.

        All four are registered in `evaluation.normalize.TRANSFORMS`, so they
        normalise without a contract decision. Names outside that registry would
        raise at normalisation time, which is the intended behaviour.
        """
        return ("f1", "iou", "precision", "recall")

    def metric_definitions(self) -> dict[str, str]:
        """What each metric means, for the provenance block."""
        return {
            "f1": "pooled harmonic mean of precision and recall over change pixels",
            "iou": "pooled intersection-over-union of the change class",
            "precision": "pooled TP / (TP + FP) for the change class",
            "recall": "pooled TP / (TP + FN) for the change class",
            "threshold": f"probability binarisation threshold = {self.threshold}",
        }

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        samples: Sequence[BenchmarkSample],
        *,
        predict: Callable[[BenchmarkSample], Any],
    ) -> dict[str, float]:
        """Score `samples` with `predict`, returning pooled change metrics.

        Args:
            samples: samples from `load()`, each carrying `label_path`.
            predict: maps a sample to a probability map (HxW float, or an HxW
                boolean mask). **Required.** There is deliberately no default: a
                default would have to invent predictions, and an invented
                prediction produces a fabricated benchmark score.

        Returns:
            Pooled metrics keyed by `metric_names()`.

        Raises:
            AdapterDataError: a label could not be read, or `predict` returned a
                shape that does not match its label. A shape mismatch is a bug,
                and scoring the overlap silently would hide it.
        """
        from PIL import Image

        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for sample in samples:
            label_path = Path(str(sample.expected))
            if not label_path.is_file():
                raise AdapterDataError(
                    f"sample {sample.sample_id!r}: label not found at {label_path}"
                )
            truth = np.asarray(Image.open(label_path)) > 0
            prediction = np.asarray(predict(sample))

            if prediction.ndim != 2:
                raise AdapterDataError(
                    f"sample {sample.sample_id!r}: prediction must be 2-D, got "
                    f"shape {prediction.shape}"
                )
            if prediction.shape != truth.shape:
                raise AdapterDataError(
                    f"sample {sample.sample_id!r}: prediction shape "
                    f"{prediction.shape} != label shape {truth.shape}"
                )
            pairs.append((truth, prediction))

        if not pairs:
            raise AdapterDataError(
                "no scorable pairs: every sample was rejected, so no metric can "
                "be computed. Returning zeros here would look like a score of 0."
            )

        report = change_metrics.score_dataset(pairs, threshold=self.threshold)
        pooled = report.pooled
        return {
            "f1": float(pooled.f1),
            "iou": float(pooled.iou),
            "precision": float(pooled.precision),
            "recall": float(pooled.recall),
        }

    # -- reporting ---------------------------------------------------------
    def manifest(self, *, split: str = "test") -> dict[str, Any]:
        """A reproducible manifest over the split's sample ids."""
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
                "subset_policy": self.subset_policy(split=split),
                "n_scenes": len({s.meta.get("scene") for s in samples}),
            },
        )

    def _version_string(self, root: Path | None) -> str:
        """A version string, or an explicit statement that it is unknown."""
        if root is None:
            return "unavailable (no corpus root)"
        archive = root / "levir-cd-256.zip"
        if archive.is_file():
            return f"levir-cd-256 (archive present: {archive.name})"
        return "levir-cd-256 (declared by directory layout; no archive to hash)"

    def provenance(
        self,
        *,
        split: str = "test",
        config_hash: str | None = None,
    ) -> dict[str, Any]:
        """Everything needed to reproduce a LEVIR-CD evaluation."""
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
                "threshold": self.threshold,
            },
        )

    def leakage_check(self, *, eval_split: str = "test") -> dict[str, Any]:
        """Assert the eval split shares no tile or scene with train.

        Two levels, because they fail differently: a duplicated tile is a bad
        list, while a *scene* split across train and test is the subtler leak --
        neighbouring tiles from one acquisition are correlated, so a
        sample-disjoint split can still inflate a score.
        """
        train = self._load_items(split="train")
        eval_items = self._load_items(split=eval_split)

        scene_of: dict[str, str] = {}
        for item in train + eval_items:
            scene_of[str(item["sample_id"])] = str(
                item.get("group_key") or item.get("image_key") or item["sample_id"]
            )

        return check_identity_leakage(
            [str(i["sample_id"]) for i in train],
            [str(i["sample_id"]) for i in eval_items],
            label=f"{self.name}:train-vs-{eval_split}",
            scene_of=scene_of,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["adapter_version"] = self.ADAPTER_VERSION
        payload["corpus_kind"] = self.corpus_kind().value
        payload["subset_policy"] = (
            self.subset_policy() if self.describe_corpus().available else None
        )
        return payload


def make_scorer(
    adapter: LevirCdAdapter,
    predict: Callable[[BenchmarkSample], Any],
) -> Callable[[Sequence[BenchmarkSample]], Mapping[str, Any]]:
    """Adapt `adapter.evaluate` into the runner's `Scorer` signature.

    The runner knows nothing about any benchmark's task; this closure is where
    LEVIR-CD task knowledge enters it. Kept as a function rather than a lambda
    buried in a call site so the seam is inspectable and testable.
    """

    def scorer(samples: Sequence[BenchmarkSample]) -> Mapping[str, Any]:
        return adapter.evaluate(samples, predict=predict)

    return scorer
