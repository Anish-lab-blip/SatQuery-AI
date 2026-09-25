"""Change-VQA benchmark adapter — the R-02 held-out pair.

A CORRECTION THAT MATTERS (measured 2026-09-23)
-----------------------------------------------
`declared.py` states, of this benchmark:

    "The one 'benchmark' whose material IS partially present: the promoted R-02
     head. But the held-out question/answer splits live in the Kaggle export,
     not in the repository, so a runner here still cannot score it."

**That is false.** The held-out splits are in the repository, at
`data/cdvqa/annotations/`, and the imagery is at `data/cdvqa/im1`, `im2`,
`label1`, `label2`. Measured:

    annotations/{Train,Val,Test,Test2}_{questions,answers,images}.json
    Train   65,967 questions / 25,600 images
    Val     16,441 questions /  6,400 images
    Test    39,686 questions / 15,488 images   <- held-out
    Test2   31,036 questions / 15,488 images   <- second held-out
    im1/im2/label1/label2   2,968 files each

    join integrity (measured, all three held-out splits):
      every question id resolves to exactly one image entry AND one answer row
      (Test: 39,686 / 39,686 / 39,686)

So this adapter is **executable**, not resource-blocked. The declared root
(`artifacts/change_vqa/run`) is where the promoted head lives, not where the
corpus lives; both are reported, and the corpus is found at its real location.

WHY THIS IS NOT A CAPTIONING BENCHMARK
--------------------------------------
CDVQA carries eight question types. Seven of them (`change_or_not`,
`increase_or_not`, `decrease_or_not`, `smallest_change`, `largest_change`,
`change_ratio`, `change_ratio_types`) are closed-form and are scored with the
plan's Change-VQA **answer accuracy**. One (`change_to_what`) is free-text and is
caption-like.

The two families are kept apart on purpose. Mixing caption metrics into the VQA
accuracy because both involve text would produce a number that is neither the
benchmark's accuracy nor its caption score. `evaluate()` therefore scores
accuracy only; `caption_types()` names the free-text types, and
`evaluate_captions()` is a separate entry point that reports a caption metric
per type -- and reports `unavailable` rather than a fake value while R-16 is
open.
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
from evaluation.metrics.vqa import answer_accuracy, normalized_exact_match
from core.code_revision import REPO_ROOT

__all__ = [
    "ChangeVqaAdapter",
    "make_scorer",
    "HELD_OUT_SPLITS",
    "CAPTION_QUESTION_TYPES",
]

#: The two held-out splits this project's R-02 run used.
HELD_OUT_SPLITS: tuple[str, ...] = ("Test", "Test2")

#: Question types whose answer is free text rather than closed-form. These are
#: caption-like and are deliberately NOT folded into the accuracy metric.
CAPTION_QUESTION_TYPES: frozenset[str] = frozenset({"change_to_what"})

#: Published split sizes, for the coverage record.
OFFICIAL_SPLIT_QUESTIONS: dict[str, int] = {
    "Train": 65_967,
    "Val": 16_441,
    "Test": 39_686,
    "Test2": 31_036,
}

_ROOT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("data/cdvqa", "discovered corpus (annotations + im1/im2/label1/label2)"),
    ("artifacts/change_vqa/run", "declared root (holds the promoted R-02 head only)"),
    ("data/change_vqa", "alternate name"),
)

#: The promoted R-02 head, recorded so provenance can name the model artifact.
R02_HEAD = REPO_ROOT / "artifacts" / "change_vqa" / "run" / "head.pt"


class ChangeVqaAdapter(BenchmarkAdapter):
    """Change-VQA over CDVQA's held-out question/answer splits."""

    name = "change_vqa_test"
    ADAPTER_VERSION = "1.0.0"

    source = BenchmarkSource(
        name="change_vqa_test",
        root=REPO_ROOT / "artifacts" / "change_vqa" / "run",
        citation=(
            "CDVQA (Change Detection VQA); first-party held-out splits as shipped "
            "in data/cdvqa/annotations/"
        ),
        note=(
            "The declared root holds the promoted R-02 head. The CORPUS is at "
            "`data/cdvqa/` (annotations + imagery), which is where this adapter "
            "reads from; both locations are reported."
        ),
    )

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        image_root: str | Path | None = None,
    ) -> None:
        self._explicit_root = Path(root) if root is not None else None
        self._explicit_image_root = (
            Path(image_root) if image_root is not None else None
        )
        self._resolved: Path | None = None
        self._locations: list = []
        self._resolved_once = False
        #: Diagnostics from the most recent `load()`, so a caller can see how many
        #: questions were dropped for want of an image record rather than having
        #: to infer it from a smaller-than-expected sample count.
        self._last_load_stats: dict[str, Any] = {}

    # -- corpus ------------------------------------------------------------
    def _resolve(self):
        if self._explicit_root is not None:
            path = self._explicit_root
            return (path if path.exists() else None), []
        if not self._resolved_once:
            self._resolved, self._locations = resolve_corpus_root(_ROOT_CANDIDATES)
            self._resolved_once = True
        return self._resolved, self._locations

    def _annotations_dir(self) -> Path | None:
        root, _ = self._resolve()
        if root is None:
            return None
        candidate = root / "annotations"
        return candidate if candidate.is_dir() else None

    def _images_root(self) -> Path | None:
        if self._explicit_image_root is not None:
            return self._explicit_image_root
        root, _ = self._resolve()
        if root is None:
            return None
        for name in ("im1", "im2"):
            if (root / name).is_dir():
                return root
        return None

    def describe_corpus(self) -> CorpusDescription:
        root, locations = self._resolve()
        if root is None:
            return CorpusDescription(
                benchmark=self.name,
                available=False,
                root=str(_ROOT_CANDIDATES[0][0]),
                reason=(
                    "no CDVQA root found. Searched: "
                    + ", ".join(str(loc.path) for loc in locations)
                ),
                note=(
                    "A CDVQA tree holds annotations/{Train,Val,Test,Test2}_"
                    "{questions,answers,images}.json plus im1/im2/label1/label2."
                ),
            )

        annotations = self._annotations_dir()
        images_root = self._images_root()

        n_pairs = 0
        if images_root is not None:
            im1 = images_root / "im1"
            if im1.is_dir():
                n_pairs = len(list(im1.glob("*")))

        n_annotation_files = (
            len(list(annotations.glob("*.json"))) if annotations else 0
        )

        reason = None
        if annotations is None:
            reason = f"no annotations/ directory under {root}"
        elif n_annotation_files == 0:
            reason = f"annotations/ exists but holds no JSON: {annotations}"

        return CorpusDescription(
            benchmark=self.name,
            available=annotations is not None and n_annotation_files > 0,
            root=str(root),
            n_files=n_annotation_files,
            bytes=0,
            kinds={
                "annotations": str(annotations or ""),
                "images": str(images_root or ""),
                "question_types": "8",
            },
            reason=reason,
            note=(
                f"resolved root: {root}; annotation JSON files: "
                f"{n_annotation_files}; image pairs on disk: {n_pairs}; "
                f"locations tried: "
                + "; ".join(f"{loc.role}={loc.path}" for loc in locations)
            ),
        )

    def is_official(self) -> bool:
        """True only when the corpus answered at the declared root.

        The declared root holds the promoted head, not the corpus, so in practice
        this is False and the adapter reports DEGRADED. That is the honest
        reading: the material is real and complete, but it is not where the
        project declared it should be.
        """
        root, _ = self._resolve()
        return root is not None and root == (REPO_ROOT / "artifacts" / "change_vqa" / "run")

    def corpus_kind(self) -> BenchmarkStatus:
        if not self.describe_corpus().available:
            return BenchmarkStatus.NOT_RUN
        return BenchmarkStatus.REAL if self.is_official() else BenchmarkStatus.DEGRADED

    # -- loading -----------------------------------------------------------
    def _load_corpus(self, splits: Sequence[str]):
        root, _ = self._resolve()
        if root is None:
            raise BenchmarkNotAvailableError(
                f"CDVQA root not found; searched "
                f"{[str(loc.path) for loc in self._locations]}"
            )
        from training.data.cdvqa import load_cdvqa

        return load_cdvqa(root, tuple(splits))

    def available_splits(self) -> tuple[str, ...]:
        """Which CDVQA splits are actually present, measured."""
        annotations = self._annotations_dir()
        if annotations is None:
            return ()
        found = []
        for split in ("Train", "Val", "Test", "Test2"):
            if (annotations / f"{split}_questions.json").is_file():
                found.append(split)
        return tuple(found)

    def _image_index(self, split_label: str) -> dict[int, dict[str, Any]]:
        """Map EVERY `image_id` to its raw record, read from the annotation file.

        This does NOT use `CDVQACorpus.images`. That list is deduplicated by
        `file_name` (measured: 968 entries for the Test split) while the questions
        reference 15,488 distinct `image_id`s that all resolve onto those 968
        files. Joining through the deduplicated list therefore silently dropped
        **35,081 of the Test split's 39,686 questions** -- the adapter reported
        4,605 samples and looked perfectly healthy doing it. Reading the raw
        table is what makes the join complete.

        Returns:
            `{image_id: {"file_name": str, "scene_key": str}}`. Empty when the
            annotation file is missing.
        """
        annotations = self._annotations_dir()
        if annotations is None:
            return {}
        path = annotations / f"{split_label}_images.json"
        if not path.is_file():
            return {}

        import json

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return {}

        records = payload.get("images") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return {}

        index: dict[int, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            try:
                image_id = int(record.get("id"))
            except (TypeError, ValueError):
                continue
            file_name = str(record.get("file_name") or "")
            index[image_id] = {
                "file_name": file_name,
                "scene_key": Path(file_name).stem if file_name else "",
                "res_x": record.get("res_x"),
                "res_y": record.get("res_y"),
            }
        return index

    def load(self, *, split: str = "test") -> list[BenchmarkSample]:
        """Package one held-out split as (question, answer, image-pair) samples.

        The split label is matched against CDVQA's own `Train|Val|Test|Test2`
        casing, resolved from the split list rather than guessed, so `test` can
        mean `Test` or `Test2` unambiguously and an unknown split raises.

        Raises:
            BenchmarkNotAvailableError: the corpus is absent, or the requested
                split is not present.
        """
        if not self.describe_corpus().available:
            description = self.describe_corpus()
            raise BenchmarkNotAvailableError(
                f"benchmark {self.name!r} is not available: "
                f"{description.reason or 'corpus not present'} "
                f"(root={description.root})",
                context=description.to_dict(),
            )

        present = self.available_splits()
        wanted = self._resolve_split_label(split, present)

        corpus = self._load_corpus([wanted])
        images_root = self._images_root()
        image_index = self._image_index(wanted)

        canonical = as_split_name(split)
        skipped_no_image = 0

        samples: list[BenchmarkSample] = []
        for question in corpus.questions:
            if question.split != wanted:
                continue
            if not question.answer:
                # A question with no resolvable answer is not evaluable. Skipped
                # rather than given a placeholder, which would score as a hit or
                # a miss against an invented gold value.
                continue

            record = image_index.get(question.image_id)
            if record is None:
                skipped_no_image += 1
                continue

            file_name = record["file_name"]
            im1 = im2 = None
            if images_root is not None and file_name:
                im1 = images_root / "im1" / file_name
                im2 = images_root / "im2" / file_name

            samples.append(
                BenchmarkSample(
                    sample_id=f"{wanted}-q{question.question_id:07d}",
                    payload={
                        "question": question.question,
                        "question_type": question.qtype,
                        "image1": str(im1) if im1 else None,
                        "image2": str(im2) if im2 else None,
                        "image1_present": bool(im1 and im1.is_file()),
                        "image2_present": bool(im2 and im2.is_file()),
                    },
                    # The gold answer, verbatim from the corpus.
                    expected=question.answer,
                    meta={
                        "split": canonical,
                        "split_label": wanted,
                        "scene": record["scene_key"],
                        "dataset": self.name,
                        "question_type": question.qtype,
                        "image_id": question.image_id,
                    },
                )
            )

        self._last_load_stats = {
            "split_label": wanted,
            "n_samples": len(samples),
            "n_skipped_no_image_record": skipped_no_image,
            "n_image_records_indexed": len(image_index),
        }
        return samples

    def last_load_stats(self) -> dict[str, Any]:
        """Diagnostics from the most recent `load()`. Public on purpose.

        Added 2026-09-23: the counts existed only as `_last_load_stats`, so the
        one signal that would reveal a bad join -- a skipped-question count above
        zero -- was reachable only by reading a private attribute. That is exactly
        the signal that had to be visible when the `file_name` join was silently
        dropping 35,081 of 39,686 Test questions, so it is now part of the
        adapter's public surface and asserted by the test suite.
        """
        return dict(self._last_load_stats)

    def _resolve_split_label(self, split: str, present: Sequence[str]) -> str:
        """Map a requested split onto CDVQA's own split label.

        Case-insensitive, and an unknown split raises rather than defaulting to
        `Test` -- defaulting would let a caller believe they evaluated the
        held-out set while actually evaluating nothing (or, worse, `Train`).
        """
        key = str(split).strip().lower()
        for candidate in present:
            if candidate.lower() == key:
                return candidate
        raise AdapterDataError(
            f"split {split!r} is not present in this CDVQA corpus. "
            f"Available splits: {list(present)}"
        )

    # -- task shape --------------------------------------------------------
    def metric_names(self) -> tuple[str, ...]:
        """`accuracy` only -- the plan's Change-VQA answer accuracy.

        Caption metrics are deliberately excluded: they apply to
        `change_to_what` alone, and folding them in here would produce a number
        that is neither the benchmark's accuracy nor its caption score.
        """
        return ("accuracy",)

    def caption_types(self) -> tuple[str, ...]:
        """Question types whose answers are free text (caption-like)."""
        return tuple(sorted(CAPTION_QUESTION_TYPES))

    def metric_definitions(self) -> dict[str, str]:
        return {
            "accuracy": (
                "fraction of questions whose prediction equals the gold answer "
                "after VQA-v2 normalization (normalized exact match)"
            ),
            "normalization": "evaluation.metrics.vqa.normalize_answer (R1-R6)",
            "caption_types_excluded": ", ".join(sorted(CAPTION_QUESTION_TYPES)),
        }

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        samples: Sequence[BenchmarkSample],
        *,
        predict: Callable[[BenchmarkSample], Any],
    ) -> dict[str, float]:
        """Answer accuracy over closed-form questions.

        Args:
            samples: samples from `load()`.
            predict: maps a sample to a predicted answer string. **Required.**

        Returns:
            `{"accuracy": float}` over the CLOSED-FORM questions. Free-text
            (`change_to_what`) samples are excluded from this figure and reported
            separately by `evaluate_by_type`, because scoring a free-text answer
            with exact match would understate it and scoring it with a caption
            metric would make the number incomparable.

        Raises:
            AdapterDataError: no scorable samples.
        """
        scored = [s for s in samples if s.meta.get("question_type") not in CAPTION_QUESTION_TYPES]
        if not scored:
            raise AdapterDataError(
                "no closed-form questions to score (all samples were free-text or "
                "the list was empty); returning 0.0 here would look like a score"
            )

        preds = [str(predict(sample)) for sample in scored]
        golds = [str(sample.expected) for sample in scored]
        return {"accuracy": float(answer_accuracy(preds, golds))}

    def evaluate_by_type(
        self,
        samples: Sequence[BenchmarkSample],
        *,
        predict: Callable[[BenchmarkSample], Any],
    ) -> dict[str, dict[str, Any]]:
        """Per-question-type accuracy, so one weak type cannot hide in an average.

        Also the honest place to see the free-text type: it is reported with its
        own count and, for `change_to_what`, a `caption_metric` field that reads
        `unavailable (R-16 open)` rather than a number.
        """
        grouped: dict[str, list[BenchmarkSample]] = {}
        for sample in samples:
            grouped.setdefault(str(sample.meta.get("question_type")), []).append(sample)

        out: dict[str, dict[str, Any]] = {}
        for qtype, group in sorted(grouped.items()):
            if qtype in CAPTION_QUESTION_TYPES:
                out[qtype] = {
                    "n": len(group),
                    "metric": "caption",
                    "caption_metric": "unavailable (R-16 open; see docs/OWNER_DECISIONS)",
                    "accuracy": None,
                    "note": (
                        "free-text answer; not scored with exact match and not "
                        "scored with a caption metric while R-16 is open"
                    ),
                }
                continue
            preds = [str(predict(s)) for s in group]
            golds = [str(s.expected) for s in group]
            out[qtype] = {
                "n": len(group),
                "metric": "answer_accuracy",
                "accuracy": float(answer_accuracy(preds, golds)),
            }
        return out

    def evaluate_captions(
        self,
        samples: Sequence[BenchmarkSample],
        *,
        predict: Callable[[BenchmarkSample], Any],
        metric: str = "rouge_l",
    ) -> dict[str, Any]:
        """Caption metrics for the free-text question types, when available.

        Routes through `evaluation.metrics.caption`, which reports a metric as
        unavailable rather than substituting another one. While R-16 is open and
        the optional dependency is absent, this returns `available: False` with
        the reason -- it does not return a fabricated score and it does not fall
        back to accuracy.
        """
        from evaluation.metrics.caption import caption_capability, score_captions

        free_text = [s for s in samples if s.meta.get("question_type") in CAPTION_QUESTION_TYPES]
        if not free_text:
            return {
                "available": False,
                "reason": "no free-text question types present in these samples",
                "caption_types": self.caption_types(),
                "metric": metric,
            }

        capability = caption_capability()
        entry = capability.get(metric)
        if entry is None:
            return {
                "available": False,
                "reason": f"unknown caption metric {metric!r}",
                "known": sorted(capability),
                "metric": metric,
            }
        if not entry["available"]:
            return {
                "available": False,
                "reason": entry["reason"],
                "implementation": entry["implementation"],
                "metric": metric,
                "n": len(free_text),
                "note": "no value is reported rather than a substitute metric",
            }

        preds = [str(predict(s)) for s in free_text]
        refs = [str(s.expected) for s in free_text]
        return {
            "available": True,
            "metric": metric,
            "n": len(free_text),
            "scores": score_captions(preds, refs, metrics=(metric,)),
            "implementation": entry["implementation"],
            "version": entry["version"],
        }

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
                "split_label": samples[0].meta["split_label"] if samples else None,
                "subset_policy": self.subset_policy(split=split),
                "n_scenes": len({s.meta.get("scene") for s in samples}),
                "question_types": sorted({str(s.meta.get("question_type")) for s in samples}),
            },
        )

    def subset_policy(self, *, split: str = "test") -> dict[str, Any]:
        """Whether the requested held-out split is present in full."""
        present = self.available_splits()
        try:
            wanted = self._resolve_split_label(split, present)
        except AdapterDataError:
            return subset_policy_block(
                is_full_corpus=False,
                n_present=0,
                n_official=None,
                basis=f"split {split!r} not present; available: {list(present)}",
            )

        n_present = len(self.load(split=split)) if present else 0
        official = OFFICIAL_SPLIT_QUESTIONS.get(wanted)
        count_complete = n_present == official

        return subset_policy_block(
            is_full_corpus=count_complete and self.is_official(),
            n_present=n_present,
            n_official=official,
            basis=(
                f"resolved (question, answer, image) triples for split {wanted!r} "
                f"were counted ({n_present}) and compared against the shipped "
                f"split size ({official}); "
                + (
                    "the corpus is at the canonical declared root"
                    if self.is_official()
                    else "the corpus is NOT at the canonical declared root"
                )
            ),
            detail=(
                "The CDVQA held-out splits ARE present in this repository at "
                "data/cdvqa/annotations/ (correcting declared.py's earlier claim "
                "that they live only in the Kaggle export). Only "
                "closed-form questions are scored with answer accuracy; "
                "free-text types are reported separately."
            ),
            extra={
                "held_out_splits_available": [
                    s for s in present if s in HELD_OUT_SPLITS
                ],
                "all_splits_available": list(present),
            },
        )

    def _version_string(self, root: Path | None) -> str:
        if root is None:
            return "unavailable (no corpus root)"
        return "CDVQA (Change Detection VQA) as shipped in data/cdvqa"

    def provenance(
        self, *, split: str = "test", config_hash: str | None = None
    ) -> dict[str, Any]:
        root, _ = self._resolve()
        artifact_hashes: dict[str, str] = {}
        if R02_HEAD.is_file():
            from evaluation.benchmark_adapters._common import sha256_file

            artifact_hashes["r02_change_vqa_head.pt"] = sha256_file(R02_HEAD)
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
                "declared_root": str(self.source.root),
                "is_official": self.is_official(),
                "caption_types": list(self.caption_types()),
            },
        )

    def leakage_check(self) -> dict[str, Any]:
        """Assert no question and no scene is shared between Train and held-out.

        Both levels are checked. A shared `scene_key` matters here because
        CDVQA's imagery is paired change data: two questions over the same
        `file_name` are the same scene, and a question-level split that reuses a
        scene would inflate the held-out figure.
        """
        present = self.available_splits()
        if "Train" not in present:
            return {
                "label": f"{self.name}:Train-vs-held-out",
                "clean": None,
                "performed": False,
                "note": "Train split absent; leakage check not performed",
            }

        corpus = self._load_corpus(list(present))

        # Scene keys come from the RAW image tables, not `corpus.images` -- that
        # list is deduplicated by file_name and would map most question ids to no
        # scene at all, quietly weakening this check.
        scene_of: dict[str, str] = {}
        for split_label in present:
            index = self._image_index(split_label)
            for question in corpus.questions:
                if question.split != split_label:
                    continue
                record = index.get(question.image_id)
                scene = record["scene_key"] if record else f"unknown-image-{question.image_id}"
                scene_of[str(question.question_id)] = scene

        train_q = [q for q in corpus.questions if q.split == "Train"]
        held_q = [q for q in corpus.questions if q.split in HELD_OUT_SPLITS]

        report = check_identity_leakage(
            [str(q.question_id) for q in train_q],
            [str(q.question_id) for q in held_q],
            label=f"{self.name}:Train-vs-{'+'.join(HELD_OUT_SPLITS)}",
            scene_of=scene_of,
        )
        report["performed"] = True
        report["n_scenes_mapped"] = len(set(scene_of.values()))
        return report

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["adapter_version"] = self.ADAPTER_VERSION
        payload["corpus_kind"] = self.corpus_kind().value
        payload["available_splits"] = list(self.available_splits())
        payload["subset_policy"] = (
            self.subset_policy() if self.describe_corpus().available else None
        )
        return payload


def make_scorer(
    adapter: ChangeVqaAdapter,
    predict: Callable[[BenchmarkSample], Any],
) -> Callable[[Sequence[BenchmarkSample]], Mapping[str, Any]]:
    """Adapt `adapter.evaluate` into the runner's `Scorer` signature."""

    def scorer(samples: Sequence[BenchmarkSample]) -> Mapping[str, Any]:
        return adapter.evaluate(samples, predict=predict)

    return scorer
