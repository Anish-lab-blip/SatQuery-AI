"""BigEarthNet-S1 benchmark adapter — multi-label SAR scene classification.

WHAT THE CORPUS ACTUALLY IS (measured 2026-09-23)
-------------------------------------------------
`data/bigearthnet_v2/reben/BigEarthNet-S1/` holds a real reBEN-format S1 tree:

    <S1A_IW_GRDH_1SDV_<ts>>/                         193 scene directories
        <S1A_IW_GRDH_1SDV_<ts>_<tile>_<row>_<col>>/ one directory per patch
            ..._VH.tif  ..._VV.tif                   the two SAR channels

Measured: **28,000 patch directories**, each with exactly the VV/VH pair.

LABELS ARE NOT PER-PATCH HERE -- THIS IS THE TRAP
-------------------------------------------------
`training.data.bigearthnet.discover_patches(..., require_labels=True)` returns
**zero** patches on this corpus and raises, because it looks for a per-patch
`*metadata*.json` and this download has none. That was verified by running it.

The labels live one level up, in a single parquet:

    data/bigearthnet_v2/metadata.parquet     480,038 rows
        patch_id, labels, split, country, s1_name, s2v1_name,
        contains_seasonal_snow, contains_cloud_or_shadow

so the adapter joins `patch_id` against `s1_name`. The join rate is measured at
call time and reported -- a silent inner join that matched nothing would look
exactly like a corpus with no labels.

MULTI-LABEL, AND IT STAYS MULTI-LABEL
-------------------------------------
`labels` is a LIST of CLC-19 class names, and many patches carry several. This
adapter scores multi-label micro-F1/precision/recall over the label sets. It
never reduces a multi-label patch to a single class: doing so would silently
change the benchmark, and the resulting number would not be comparable to any
published BigEarthNet figure.

SUBSET, DISCLOSED
-----------------
28,000 patches are present against 480,038 in the official metadata. That is
5.8% of the benchmark, so the adapter reports `DEGRADED` with an explicit subset
policy. The numbers it produces are real, and they describe 28,000 patches --
not BigEarthNet.
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
from core.code_revision import REPO_ROOT

__all__ = ["BigEarthNetS1Adapter", "make_scorer", "OFFICIAL_PATCH_COUNT"]

#: Rows in the official metadata table, for the coverage record.
OFFICIAL_PATCH_COUNT = 480_038

#: The CLC-19 class set, imported lazily in `class_names()` so this module stays
#: importable without the training package's numpy stack at import time.

_ROOT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("data/BigEarthNet-S1", "canonical target (declared.py)"),
    ("data/bigearthnet_v2/reben/BigEarthNet-S1", "discovered corpus (reBEN layout)"),
)

#: Where the label table lives, relative to the reBEN root's parent.
_METADATA_CANDIDATES: tuple[str, ...] = (
    "data/bigearthnet_v2/metadata.parquet",
    "data/bigearthnet_v2/reben/metadata.parquet",
    "metadata.parquet",
)

#: SAR channels this benchmark uses.
SAR_CHANNELS: tuple[str, ...] = ("VV", "VH")


def _coerce_labels(value: Any) -> tuple[str, ...]:
    """Read one row's label cell into a tuple of class names.

    Three representations are handled, all of which were considered because
    getting this wrong silently changes the benchmark:

      1. a sequence (the measured case -- pandas returns a numpy `ndarray` of
         `object` dtype). Iterated directly.
      2. a string holding a bracketed list, e.g. `"['Arable land', 'Pastures']"`,
         as produced when a parquet/CSV round-trip stringifies a list column.
         Parsed with `ast.literal_eval`.
      3. a plain string, e.g. `"Arable land"`, which is genuinely ONE label.

    Case 2 is the dangerous one: treating a stringified list as a single label
    yields `("['Arable land', 'Pastures']",)` -- one nonsense class -- and the
    metric would then score every patch against a label that does not exist. It
    is parsed, and a bracketed string that will NOT parse raises rather than
    being accepted as a single label.

    HARDENED 2026-09-23: the bracket test was `startswith("[") and endswith("]")`,
    so a stringified list that had LOST its closing bracket -- exactly the shape a
    truncating CSV writer or a manual edit produces -- failed the test and fell
    through to case 3, where it was accepted as one label. That is the very
    outcome case 2 exists to prevent, reached by a one-character corruption. The
    test is now "starts with `[` OR ends with `]`": any string that looks like a
    bracketed list on either end is parsed, and raises if it will not parse.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.endswith("]"):
            import ast

            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError) as exc:
                raise AdapterDataError(
                    f"label cell {text[:80]!r} looks like a stringified list but "
                    f"does not parse ({exc}); refusing to treat it as a single "
                    f"label, which would invent a class name"
                ) from exc
            if isinstance(parsed, (list, tuple)):
                return tuple(str(v) for v in parsed)
        return (text,) if text else ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(v) for v in value)
    # numpy ndarray and any other iterable of strings.
    try:
        return tuple(str(v) for v in value)
    except TypeError:
        return (str(value),)


def _normalise_split(raw: str) -> str:
    """Map a metadata split label onto train|val|test.

    The measured table uses `validation`, not `val`. Without this mapping the
    val split would match nothing and silently evaluate zero samples -- which
    reads as "the corpus has no val data" rather than "the labels did not line
    up". Delegates to `as_split_name` so there is one alias table in the project.
    """
    return as_split_name(raw)


class BigEarthNetS1Adapter(BenchmarkAdapter):
    """Multi-label SAR classification on a disclosed BigEarthNet-S1 subset."""

    name = "bigearthnet_s1"
    ADAPTER_VERSION = "1.0.0"

    source = BenchmarkSource(
        name="bigearthnet_s1",
        root=REPO_ROOT / "data" / "BigEarthNet-S1",
        citation="BigEarthNet-S1, as recorded in the implementation plan",
        note=(
            "Canonical target root. The corpus is present at "
            "`data/bigearthnet_v2/reben/BigEarthNet-S1` rather than here; labels "
            "come from `data/bigearthnet_v2/metadata.parquet`, not per-patch JSON."
        ),
    )

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        metadata_path: str | Path | None = None,
        score_threshold: float = 0.5,
    ) -> None:
        self._explicit_root = Path(root) if root is not None else None
        self._explicit_metadata = (
            Path(metadata_path) if metadata_path is not None else None
        )
        self.score_threshold = float(score_threshold)
        self._resolved: Path | None = None
        self._locations: list = []
        self._resolved_once = False
        self._patches_cache: list | None = None
        self._labels_cache: dict[str, tuple[str, ...]] | None = None
        self._splits_cache: dict[str, str] | None = None

    # -- corpus ------------------------------------------------------------
    def _resolve(self):
        if self._explicit_root is not None:
            path = self._explicit_root
            return (path if path.exists() else None), []
        if not self._resolved_once:
            self._resolved, self._locations = resolve_corpus_root(_ROOT_CANDIDATES)
            self._resolved_once = True
        return self._resolved, self._locations

    def _metadata_file(self) -> Path | None:
        if self._explicit_metadata is not None:
            return self._explicit_metadata if self._explicit_metadata.is_file() else None
        for candidate in _METADATA_CANDIDATES:
            path = REPO_ROOT / candidate
            if path.is_file():
                return path
        return None

    def _patches(self) -> list:
        """Every S1 patch on disk. Cached; the walk is not cheap."""
        if self._patches_cache is not None:
            return self._patches_cache
        root, _ = self._resolve()
        if root is None:
            return []
        from training.data.bigearthnet import discover_patches

        # require_labels=False is mandatory here: the labels are not per-patch in
        # this download, and require_labels=True raises on the whole corpus.
        self._patches_cache = list(
            discover_patches(root, "s1", require_labels=False)
        )
        return self._patches_cache

    def _label_table(self) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
        """Join `metadata.parquet` into `{patch_id: labels}` and `{patch_id: split}`.

        Returns empty dicts (never raises) when the table is missing, so
        `describe_corpus()` can report the absence rather than crashing on it.
        """
        if self._labels_cache is not None and self._splits_cache is not None:
            return self._labels_cache, self._splits_cache

        table = self._metadata_file()
        if table is None:
            self._labels_cache, self._splits_cache = {}, {}
            return self._labels_cache, self._splits_cache

        try:
            import pandas as pd
        except ImportError:
            self._labels_cache, self._splits_cache = {}, {}
            return self._labels_cache, self._splits_cache

        frame = pd.read_parquet(table, columns=["s1_name", "labels", "split"])
        labels: dict[str, tuple[str, ...]] = {}
        splits: dict[str, str] = {}
        for name, value, split in zip(
            frame["s1_name"], frame["labels"], frame["split"]
        ):
            key = str(name)
            labels[key] = _coerce_labels(value)
            splits[key] = _normalise_split(str(split))

        self._labels_cache, self._splits_cache = labels, splits
        return labels, splits

    def describe_corpus(self) -> CorpusDescription:
        root, locations = self._resolve()
        if root is None:
            return CorpusDescription(
                benchmark=self.name,
                available=False,
                root=str(_ROOT_CANDIDATES[0][0]),
                reason=(
                    "no BigEarthNet-S1 root found. Searched: "
                    + ", ".join(str(loc.path) for loc in locations)
                ),
                note="Expected <root>/<scene>/<patch>/{..._VV.tif,..._VH.tif}.",
            )

        patches = self._patches()
        labels, _ = self._label_table()
        table = self._metadata_file()
        n_labelled = sum(1 for p in patches if labels.get(str(p.patch_id)))

        reason = None
        if not patches:
            reason = f"root exists but no S1 patch directories were found: {root}"
        elif table is None:
            reason = (
                f"{len(patches)} patches found but the label table "
                f"(metadata.parquet) is absent, so no ground truth is available"
            )

        return CorpusDescription(
            benchmark=self.name,
            available=bool(patches) and table is not None,
            root=str(root),
            n_files=len(patches),
            bytes=0,
            kinds={"channels": ",".join(SAR_CHANNELS), "labels": "multi-label CLC-19"},
            reason=reason,
            note=(
                f"resolved root: {root}; patches: {len(patches)}; "
                f"label table: {table}; patches with labels: {n_labelled}"
            ),
        )

    def is_official(self) -> bool:
        root, _ = self._resolve()
        return root is not None and root == (REPO_ROOT / "data" / "BigEarthNet-S1")

    def corpus_kind(self) -> BenchmarkStatus:
        if not self.describe_corpus().available:
            return BenchmarkStatus.NOT_RUN
        return BenchmarkStatus.REAL if self.is_official() else BenchmarkStatus.DEGRADED

    def subset_policy(self, *, split: str = "test") -> dict[str, Any]:
        """The explicit subset record: how much of BigEarthNet is present here.

        Carries three measured facts a reader needs to interpret any number this
        adapter produces, because each one independently bounds what the number
        can mean:

          * **coverage** -- 28,000 of 480,038 official patches (5.8%).
          * **join rate** -- some on-disk patches have no row in the label table,
            so they are not evaluable and are excluded.
          * **label cardinality** -- measured over the present patches. This
            download happens to be entirely single-label, while the official
            table is genuinely multi-label (1-11 labels; only 18% single-label).
            So a score from this subset CANNOT be compared to a published
            multi-label BigEarthNet figure, and saying so here is the difference
            between a bounded number and a misleading one.
        """
        patches = self._patches()
        labels, splits = self._label_table()

        present_ids = [str(p.patch_id) for p in patches]
        matched = [k for k in present_ids if labels.get(k)]
        unmatched = len(present_ids) - len(matched)

        cardinality: dict[int, int] = {}
        for key in matched:
            n = len(labels[key])
            cardinality[n] = cardinality.get(n, 0) + 1

        n_split = sum(1 for k in matched if splits.get(k) == as_split_name(split))
        multi_label_present = sum(
            count for size, count in cardinality.items() if size > 1
        )

        return subset_policy_block(
            is_full_corpus=False,  # measured below; never True for this download
            n_present=len(patches),
            n_official=OFFICIAL_PATCH_COUNT,
            basis=(
                "the number of patch directories on disk was counted and compared "
                "against the row count of the official metadata table; the label "
                "join rate and the label-cardinality distribution were measured "
                "over those patches"
            ),
            detail=(
                f"{len(patches)} of {OFFICIAL_PATCH_COUNT} official patches are "
                f"present ({len(patches) / OFFICIAL_PATCH_COUNT:.1%}). "
                f"Label join: {len(matched)} matched, {unmatched} on-disk patches "
                f"have no row in the label table and are excluded as unevaluable. "
                f"{n_split} matched patches carry split={as_split_name(split)!r}. "
                f"Label cardinality over the present patches: "
                f"{dict(sorted(cardinality.items()))} -- "
                f"{multi_label_present} patches with more than one label. "
                f"The official table is multi-label (1-11 labels); this download "
                f"is effectively single-label, so a score from it must NOT be "
                f"compared to a published multi-label BigEarthNet figure."
            ),
            extra={
                "n_matched_in_label_table": len(matched),
                "n_unmatched_in_label_table": unmatched,
                "label_cardinality": {str(k): v for k, v in sorted(cardinality.items())},
                "n_multi_label_present": multi_label_present,
                "official_label_cardinality_note": (
                    "official table: 1-11 labels per patch; only 86,375 of "
                    "480,038 (18.0%) are single-label"
                ),
            },
        )

    # -- loading -----------------------------------------------------------
    def load(self, *, split: str = "test") -> list[BenchmarkSample]:
        """Package the split's patches, carrying BAND PATHS and LABEL SETS."""
        description = self.describe_corpus()
        if not description.available:
            raise BenchmarkNotAvailableError(
                f"benchmark {self.name!r} is not available: "
                f"{description.reason or 'corpus not present'} "
                f"(root={description.root})",
                context=description.to_dict(),
            )

        canonical = as_split_name(split)
        labels, splits = self._label_table()

        samples: list[BenchmarkSample] = []
        for patch in self._patches():
            key = str(patch.patch_id)
            if splits.get(key) != canonical:
                continue
            patch_labels = labels.get(key, ())
            if not patch_labels:
                # A patch with no labels is not an evaluable sample. It is skipped
                # rather than given an empty label set, because an empty set would
                # score as "predicted nothing, truth nothing" -- a free hit.
                continue

            bands = {name: str(path) for name, path in (patch.band_paths or {}).items()}
            missing = [c for c in SAR_CHANNELS if c not in bands]
            if missing:
                continue

            samples.append(
                BenchmarkSample(
                    sample_id=key,
                    payload={"band_paths": bands, "channels": list(SAR_CHANNELS)},
                    # Multi-label: a tuple of class names, never a single class.
                    expected=tuple(patch_labels),
                    meta={
                        "split": canonical,
                        "split_label": split,
                        "scene": str(patch.tile),
                        "dataset": self.name,
                        "country": None,
                        "n_labels": len(patch_labels),
                    },
                )
            )
        return samples

    # -- task shape --------------------------------------------------------
    def metric_names(self) -> tuple[str, ...]:
        """Micro-averaged multi-label metrics, all registered in `normalize`."""
        return ("f1", "precision", "recall")

    def class_names(self) -> tuple[str, ...]:
        """The CLC-19 label space, from the training package's own definition."""
        from training.data.bigearthnet import CLC19_CLASSES

        return tuple(CLC19_CLASSES)

    def metric_definitions(self) -> dict[str, str]:
        return {
            "f1": "micro-averaged F1 over the multi-label CLC-19 label sets",
            "precision": "micro-averaged precision over the label sets",
            "recall": "micro-averaged recall over the label sets",
            "averaging": "micro (per-label-decision), not macro and not per-sample",
            "score_threshold": (
                f"per-class score >= {self.score_threshold} counts as predicted; "
                f"ignored when the predictor returns class names"
            ),
            "label_space": "CLC-19 (19 classes)",
        }

    # -- evaluation --------------------------------------------------------
    def _as_label_set(self, prediction: Any) -> set[str]:
        """Coerce a prediction to a set of class names.

        Accepts either an explicit collection of names, or a per-class score
        vector of length 19. Anything else is rejected -- guessing at a
        prediction's meaning is how a wrong number gets produced.
        """
        if isinstance(prediction, (set, frozenset)):
            return {str(v) for v in prediction}
        if isinstance(prediction, (list, tuple)):
            if not prediction:
                return set()
            if all(isinstance(v, str) for v in prediction):
                return {str(v) for v in prediction}
            classes = self.class_names()
            if len(prediction) != len(classes):
                raise AdapterDataError(
                    f"numeric prediction has {len(prediction)} entries but the "
                    f"label space has {len(classes)}; cannot map scores to classes"
                )
            return {
                name
                for name, score in zip(classes, prediction)
                if float(score) >= self.score_threshold
            }
        raise AdapterDataError(
            f"prediction must be a collection of class names or a score vector, "
            f"got {type(prediction).__name__}"
        )

    def evaluate(
        self,
        samples: Sequence[BenchmarkSample],
        *,
        predict: Callable[[BenchmarkSample], Any],
    ) -> dict[str, float]:
        """Micro-averaged multi-label precision/recall/F1.

        Micro, not macro: every label decision counts equally, so a rare class
        contributes in proportion to how often it actually occurs. Macro would
        let a single rare class move the headline number.

        Raises:
            AdapterDataError: no samples, or a prediction that cannot be read as
                a label set.
        """
        if not samples:
            raise AdapterDataError(
                "no samples to score; returning 0.0 here would look like a score"
            )

        tp = fp = fn = 0
        for sample in samples:
            truth = {str(v) for v in (sample.expected or ())}
            predicted = self._as_label_set(predict(sample))
            tp += len(predicted & truth)
            fp += len(predicted - truth)
            fn += len(truth - predicted)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return {"f1": f1, "precision": precision, "recall": recall}

    # -- reporting ---------------------------------------------------------
    def manifest(self, *, split: str = "test") -> dict[str, Any]:
        samples = self.load(split=split)
        root, _ = self._resolve()
        table = self._metadata_file()
        return sample_manifest(
            dataset=self.name,
            version=self._version_string(root),
            split=as_split_name(split),
            sample_ids=[s.sample_id for s in samples],
            root=root,
            extra={
                "split_label": split,
                "label_table": str(table) if table else None,
                "subset_policy": self.subset_policy(split=split),
                "n_scenes": len({s.meta.get("scene") for s in samples}),
                "n_labels_total": sum(s.meta.get("n_labels", 0) for s in samples),
            },
        )

    def _version_string(self, root: Path | None) -> str:
        if root is None:
            return "unavailable (no corpus root)"
        return (
            "BigEarthNet v2 reBEN S1 subset (labels: metadata.parquet, "
            "CLC-19 multi-label); revision not pinned in-repo"
        )

    def provenance(
        self, *, split: str = "test", config_hash: str | None = None
    ) -> dict[str, Any]:
        root, _ = self._resolve()
        table = self._metadata_file()
        artifact_hashes: dict[str, str] = {}
        if table is not None:
            from evaluation.benchmark_adapters._common import sha256_file

            artifact_hashes["metadata.parquet"] = sha256_file(table)
        return provenance_block(
            adapter=type(self).__name__,
            adapter_version=self.ADAPTER_VERSION,
            dataset=self.name,
            version=self._version_string(root),
            split=as_split_name(split),
            manifest=self.manifest(split=split),
            config_hash=config_hash,
            artifact_hashes=artifact_hashes,
            metric_definitions=self.metric_definitions(),
            extra={
                "resolved_root": str(root) if root else None,
                "is_official": self.is_official(),
                "channels": list(SAR_CHANNELS),
                "label_space_size": 19,
            },
        )

    def leakage_check(self, *, eval_split: str = "test") -> dict[str, Any]:
        """Assert no patch -- and no SCENE -- is shared between train and eval.

        The scene-level check matters most here: patches from one Sentinel
        acquisition overlap geographically, so a patch-disjoint split can still be
        scene-leaky. `discover_patches` resolves the tile directory, which is the
        scene key.
        """
        labels, splits = self._label_table()
        patches = self._patches()

        train_ids, train_scenes = [], []
        eval_ids, eval_scenes = [], []
        for patch in patches:
            key = str(patch.patch_id)
            if not labels.get(key):
                continue
            if splits.get(key) == "train":
                train_ids.append(key)
                train_scenes.append(str(patch.tile))
            elif splits.get(key) == as_split_name(eval_split):
                eval_ids.append(key)
                eval_scenes.append(str(patch.tile))

        report = check_identity_leakage(
            train_ids,
            eval_ids,
            label=f"{self.name}:train-vs-{eval_split}",
        )
        shared_scenes = sorted(set(train_scenes) & set(eval_scenes))
        report["n_shared_scenes"] = len(shared_scenes)
        report["shared_scenes"] = shared_scenes[:20]
        report["scene_check_performed"] = True
        report["clean"] = report["clean"] and not shared_scenes
        if shared_scenes:
            from core.errors import LeakageError

            raise LeakageError(
                f"{self.name}: {len(shared_scenes)} Sentinel tile(s) appear in both "
                f"train and {eval_split}, e.g. {shared_scenes[:5]}",
                context=report,
            )
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
    adapter: BigEarthNetS1Adapter,
    predict: Callable[[BenchmarkSample], Any],
) -> Callable[[Sequence[BenchmarkSample]], Mapping[str, Any]]:
    """Adapt `adapter.evaluate` into the runner's `Scorer` signature."""

    def scorer(samples: Sequence[BenchmarkSample]) -> Mapping[str, Any]:
        return adapter.evaluate(samples, predict=predict)

    return scorer
