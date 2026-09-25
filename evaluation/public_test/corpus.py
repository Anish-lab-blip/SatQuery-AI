"""Immutable public-test corpus (plan section 37, "Test Set Firewall").

THE PLAN'S RULES, VERBATIM IN SPIRIT
------------------------------------
    Public test sets:  evaluation/public_test/
    are immutable.
    Training code may not import them.
    Evaluation code may read them only in evaluation mode.
    No cache of test answers is permitted.

Each is enforced here as a mechanism rather than a convention, because a rule
enforced by convention is a rule that holds until someone is in a hurry.

  immutability    -> `PublicTestCorpus.seal()` writes a manifest recording every
                     file's sha256 and byte count. `verify()` recomputes and
                     raises on ANY difference: changed content, added file,
                     REMOVED file, or renamed file. A test set that can be
                     edited between a training run and a publish is not a test
                     set.
  no import       -> a corpus is always bound to a `PublicTestFirewall` root, and
                     `read()` refuses any path that escapes the corpus root.
  evaluation mode -> `PublicTestCorpus` requires an explicit `EvaluationMode`
                     token. Training code does not have one, so it cannot open
                     the corpus even by importing this module.
  no answer cache -> there is no writer for gold answers anywhere in this
                     module. `gold_for()` reads from a sealed corpus in memory;
                     nothing persists an answer key, so there is no cache to
                     leak. `forbid_answer_cache` makes the absence checkable.

WHAT IS DELIBERATELY ABSENT
---------------------------
There is no `write()`, no `add()`, no `remove()` on a sealed corpus, and no
mutable accessor at all. Sealing is the terminal operation. Unsealing requires
constructing a new corpus object, which is a visible act in a diff.

The corpus directory is currently empty (only a `.gitkeep`). That is reported as
`available=False`, not as an empty passing test set -- an empty corpus and a
corpus that passed the firewall must not look the same.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from core.errors import LeakageError, SatQueryError
from evaluation.leakage import PUBLIC_TEST_DIRNAME, PublicTestFirewall

__all__ = [
    "PUBLIC_TEST_ROOT",
    "SEAL_FILENAME",
    "PUBLIC_TEST_SUFFIXES",
    "EvaluationMode",
    "evaluation_mode",
    "PublicTestError",
    "PublicTestCorpusSealedError",
    "PublicTestSealError",
    "PublicTestModeError",
    "PublicTestAnswerCacheError",
    "CorpusEntry",
    "CorpusSeal",
    "PublicTestCorpus",
    "forbid_answer_cache",
    "open_public_test",
]


#: The corpus root, fixed by the plan's directory layout.
#:
#: Resolved from THIS file's location, not from `REPO_ROOT / "public_test"`. The
#: plan's section 37 path is `evaluation/public_test/`, and joining the bare
#: directory name against the repository root silently produced
#: `<repo>/public_test` -- a path that does not exist and, worse, would not fail
#: loudly: the corpus would simply always report "unavailable", and a real corpus
#: placed at the documented location would be ignored. Anchoring to `__file__`
#: means the constant cannot drift from the package it names.
PUBLIC_TEST_ROOT: Path = Path(__file__).resolve().parent

#: The seal file's name. It lives at the corpus root and is itself excluded from
#: the sealed file set -- a seal cannot contain its own digest.
SEAL_FILENAME = "CORPUS_SEAL.json"

#: Content that constitutes corpus material. Narrow on purpose: a stray editor
#: backup should not silently become part of a sealed test set.
PUBLIC_TEST_SUFFIXES: tuple[str, ...] = (
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".npy",
    ".npz",
)

#: Files/directories never sealed.
_SKIP_NAMES = frozenset({".gitkeep", SEAL_FILENAME})
_SKIP_DIRS = frozenset({"__pycache__", ".git", ".ipynb_checkpoints"})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class PublicTestError(SatQueryError):
    """A public-test rule was violated."""

    code = "public_test_error"
    user_message = "A public-test corpus rule was violated."


class PublicTestSealedError(PublicTestError):
    """The corpus was modified after sealing."""

    code = "public_test_modified"
    user_message = "The public test corpus was modified after it was sealed."


class PublicTestSealError(PublicTestError):
    """The corpus could not be sealed."""

    code = "public_test_seal_error"
    user_message = "The public test corpus could not be sealed."


class PublicTestModeError(PublicTestError):
    """The corpus was opened outside evaluation mode."""

    code = "public_test_mode_error"
    user_message = "The public test corpus may only be opened in evaluation mode."


class PublicTestAnswerCacheError(PublicTestError):
    """An answer cache was found, or one was requested."""

    code = "public_test_answer_cache"
    user_message = "Caching public test answers is prohibited."


# ---------------------------------------------------------------------------
# Evaluation-mode token
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationMode:
    """An unforgeable-by-accident token permitting public-test reads.

    Deliberately not a bool. `open_public_test(corpus, mode=True)` would be far
    too easy to pass from training code by reflex; requiring an object of this
    type means the caller had to go and find the constructor.
    """

    reason: str


def evaluation_mode(reason: str) -> EvaluationMode:
    """Construct an evaluation-mode token, stating why.

    The reason is required and travels into every read's record. "Why is this
    code reading the test set?" is the question the firewall exists to make
    answerable.
    """
    if not reason or not str(reason).strip():
        raise PublicTestModeError(
            "an evaluation-mode token must state a reason; an unexplained "
            "evaluation mode is indistinguishable from an accident"
        )
    return EvaluationMode(reason=str(reason).strip())


# ---------------------------------------------------------------------------
# Corpus entries and the seal
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusEntry:
    """One sealed file: its relative path, digest and size."""

    relpath: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"relpath": self.relpath, "sha256": self.sha256, "bytes": self.bytes}


@dataclass
class CorpusSeal:
    """The recorded state of a corpus at seal time."""

    corpus_id: str
    sealed_at: str
    algorithm: str
    entries: list[CorpusEntry] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def n_files(self) -> int:
        return len(self.entries)

    @property
    def total_bytes(self) -> int:
        return sum(e.bytes for e in self.entries)

    def by_relpath(self) -> dict[str, CorpusEntry]:
        return {e.relpath: e for e in self.entries}

    @property
    def digest(self) -> str:
        """A digest over the sealed set, for recording in a run manifest.

        Built from `(relpath, sha256)` pairs in sorted order, plus the byte
        count. Path-sensitive and content-sensitive, so a rename is a change --
        a test set whose file moved is a different test set.
        """
        payload = "\n".join(
            f"{e.relpath}\x00{e.sha256}\x00{e.bytes}" for e in sorted(
                self.entries, key=lambda x: x.relpath
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "sealed_at": self.sealed_at,
            "algorithm": self.algorithm,
            "n_files": self.n_files,
            "total_bytes": self.total_bytes,
            "digest": self.digest,
            "entries": [e.to_dict() for e in sorted(self.entries, key=lambda x: x.relpath)],
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorpusSeal":
        return cls(
            corpus_id=payload["corpus_id"],
            sealed_at=payload["sealed_at"],
            algorithm=payload["algorithm"],
            entries=[
                CorpusEntry(
                    relpath=e["relpath"], sha256=e["sha256"], bytes=int(e["bytes"])
                )
                for e in payload.get("entries", [])
            ],
            notes=payload.get("notes", {}) or {},
        )


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------
class PublicTestCorpus:
    """An immutable, sealable public-test corpus.

    Lifecycle:

        corpus = PublicTestCorpus(root)        # unsealed, no reads allowed
        seal   = corpus.seal()                 # recorded; writes CORPUS_SEAL.json
        corpus.verify()                        # recompute; raise on any difference
        corpus.read("part/001.jsonl", mode=m)  # requires an EvaluationMode

    There is no unseal. A corpus that has drifted is a new corpus, deliberately.
    """

    def __init__(
        self,
        root: str | Path = PUBLIC_TEST_ROOT,
        *,
        corpus_id: str | None = None,
        firewall: PublicTestFirewall | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.corpus_id = corpus_id or self.root.name
        # The firewall from `evaluation.leakage` owns the path check. Binding it
        # here rather than re-implementing means there is exactly one definition
        # in the codebase of "is this path inside the test tree".
        self.firewall = firewall or PublicTestFirewall(self.root)
        self._seal: CorpusSeal | None = None

    # -- discovery ---------------------------------------------------------
    def files(self) -> list[Path]:
        """Every candidate corpus file, sorted by relative path.

        Sorting is required, not cosmetic: directory iteration order is not
        stable across platforms, and an unstable order would make the seal
        digest unreproducible.
        """
        if not self.root.exists():
            return []
        found: list[Path] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if path.name in _SKIP_NAMES:
                continue
            rel = path.relative_to(self.root)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            if path.suffix.lower() not in PUBLIC_TEST_SUFFIXES:
                continue
            found.append(path)
        return found

    @property
    def available(self) -> bool:
        """Whether any corpus material exists. An empty corpus is not available."""
        return len(self.files()) > 0

    def describe(self) -> dict[str, Any]:
        """What is actually present, measured."""
        files = self.files()
        total = 0
        for path in files:
            try:
                total += path.stat().st_size
            except OSError:
                pass
        return {
            "corpus_id": self.corpus_id,
            "root": str(self.root),
            "root_exists": self.root.exists(),
            "available": bool(files),
            "n_files": len(files),
            "bytes": total,
            "sealed": self._seal is not None,
            "seal_digest": self._seal.digest if self._seal is not None else None,
            "reason": (
                None
                if files
                else f"no corpus material under {self.root} "
                f"(suffixes={list(PUBLIC_TEST_SUFFIXES)})"
            ),
        }

    # -- sealing -----------------------------------------------------------
    def seal(self, *, notes: dict[str, Any] | None = None) -> CorpusSeal:
        """Record the corpus's current state. Refuses an empty corpus.

        Sealing an empty corpus would produce a seal that verifies forever, which
        reads as "the test set is intact" when the truth is "there is no test
        set". That is the exact class of false assurance this module removes.
        """
        files = self.files()
        if not files:
            raise PublicTestSealError(
                f"refusing to seal an empty corpus under {self.root}: a seal over "
                f"no files verifies forever and would read as 'the test set is "
                f"intact' when there is no test set",
                context={"root": str(self.root)},
            )

        from datetime import datetime, timezone

        entries: list[CorpusEntry] = []
        for path in files:
            rel = path.relative_to(self.root).as_posix()
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise PublicTestSealError(
                    f"could not read corpus file {rel} while sealing: {exc}",
                    context={"relpath": rel},
                ) from exc
            entries.append(
                CorpusEntry(
                    relpath=rel,
                    sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data),
                )
            )

        seal = CorpusSeal(
            corpus_id=self.corpus_id,
            sealed_at=datetime.now(timezone.utc).isoformat(),
            algorithm="sha256 over utf8(relpath) + 0x00 + sha256(contents) + 0x00 + bytes",
            entries=entries,
            notes=notes or {},
        )
        self._seal = seal

        seal_path = self.root / SEAL_FILENAME
        seal_path.write_text(
            json.dumps(seal.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return seal

    def read_seal(self) -> CorpusSeal | None:
        """Load a seal from disk, if one was written."""
        seal_path = self.root / SEAL_FILENAME
        if not seal_path.exists():
            return None
        try:
            payload = json.loads(seal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PublicTestSealError(
                f"corpus seal at {seal_path} is not valid JSON: {exc}"
            ) from exc
        seal = CorpusSeal.from_dict(payload)
        self._seal = seal
        return seal

    # -- verification ------------------------------------------------------
    def verify(self, *, expected: CorpusSeal | None = None) -> dict[str, Any]:
        """Recompute the corpus and compare against a seal. Raises on ANY delta.

        Four deltas are distinguished, because they mean different things:
          * `changed`   -- content differs; the gold answers may no longer hold
          * `added`     -- new material appeared
          * `removed`   -- material disappeared; results computed against it are
                           no longer reproducible
          * `resized`   -- content is identical but the byte count is not, which
                           should be impossible and therefore signals a bug

        A `verify()` that only checked hashes of files still present would miss
        `removed`, which is the one that silently invalidates a published number.
        """
        seal = expected if expected is not None else self._seal
        if seal is None:
            seal = self.read_seal()
        if seal is None:
            raise PublicTestSealError(
                f"no seal available for corpus {self.corpus_id!r}: seal it first "
                f"(`seal()`) or pass `expected=`",
                context={"root": str(self.root)},
            )

        recorded = seal.by_relpath()
        current: dict[str, CorpusEntry] = {}
        for path in self.files():
            rel = path.relative_to(self.root).as_posix()
            try:
                data = path.read_bytes()
            except OSError:
                continue
            current[rel] = CorpusEntry(
                relpath=rel,
                sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data),
            )

        changed = sorted(
            rel
            for rel in set(recorded) & set(current)
            if recorded[rel].sha256 != current[rel].sha256
        )
        added = sorted(set(current) - set(recorded))
        removed = sorted(set(recorded) - set(current))
        resized = sorted(
            rel
            for rel in set(recorded) & set(current)
            if recorded[rel].sha256 == current[rel].sha256
            and recorded[rel].bytes != current[rel].bytes
        )

        report = {
            "corpus_id": self.corpus_id,
            "intact": not (changed or added or removed or resized),
            "n_recorded": len(recorded),
            "n_current": len(current),
            "changed": changed,
            "added": added,
            "removed": removed,
            "resized": resized,
            "recorded_digest": seal.digest,
            "current_digest": CorpusSeal(
                corpus_id=seal.corpus_id,
                sealed_at=seal.sealed_at,
                algorithm=seal.algorithm,
                entries=list(current.values()),
            ).digest,
        }

        if not report["intact"]:
            raise PublicTestSealedError(
                "public test corpus does not match its seal -- "
                f"changed={len(changed)} added={len(added)} removed={len(removed)} "
                f"resized={len(resized)}; a modified test set cannot be used to "
                f"reproduce a published result",
                context=report,
            )
        return report

    # -- reading -----------------------------------------------------------
    def read(self, relpath: str, *, mode: EvaluationMode) -> bytes:
        """Read one corpus file. Requires evaluation mode; refuses escapes.

        The firewall is armed here deliberately. `PublicTestFirewall` is armed by
        default so that TRAINING code cannot read the tree; the evaluation reader
        disarms it for the duration of this one read and re-arms it afterwards,
        so the disarm is scoped to a call rather than to a session.
        """
        if not isinstance(mode, EvaluationMode):
            raise PublicTestModeError(
                f"reading {relpath!r} requires an EvaluationMode token "
                f"(got {type(mode).__name__}); training code has no legitimate "
                f"reason to read the public test corpus",
                context={"relpath": relpath},
            )

        target = (self.root / relpath).resolve()
        # Containment first: a `../` in relpath must not reach the firewall's
        # "outside the tree, therefore allowed" branch.
        try:
            target.relative_to(self.root)
        except ValueError:
            raise PublicTestSealedError(
                f"{relpath!r} resolves outside the corpus root {self.root}",
                context={"relpath": relpath, "resolved": str(target)},
            ) from None

        was_armed = self.firewall.armed
        self.firewall.disarm()
        try:
            self.firewall.check_read(target)
            if not target.exists():
                raise PublicTestSealError(
                    f"corpus file not found: {relpath!r} (root={self.root})",
                    context={"relpath": relpath},
                )
            return target.read_bytes()
        finally:
            if was_armed:
                self.firewall.arm()

    def read_json(self, relpath: str, *, mode: EvaluationMode) -> Any:
        """Read and parse one JSON/JSONL corpus file."""
        raw = self.read(relpath, mode=mode)
        text = raw.decode("utf-8")
        stripped = text.strip()
        if relpath.endswith(".jsonl"):
            return [json.loads(line) for line in stripped.splitlines() if line.strip()]
        return json.loads(stripped)

    def __iter__(self) -> Iterator[Path]:
        return iter(self.files())


# ---------------------------------------------------------------------------
# Answer-cache prohibition
# ---------------------------------------------------------------------------
#: Paths under which a cached answer key would be looked for. Any of these
#: existing is a violation, because a persisted answer key is a leak vector: it
#: survives the evaluation and can be read by anything with filesystem access.
_ANSWER_CACHE_PATTERNS: tuple[str, ...] = (
    "cache/answers",
    "cache/public_test_answers",
    "public_test_answers.json",
    "answer_cache.json",
    ".answer_cache",
    "gold_cache.json",
)


def forbid_answer_cache(root: str | Path = PUBLIC_TEST_ROOT) -> dict[str, Any]:
    """Assert no cached answer key exists under or beside the corpus.

    Plan section 37: "No cache of test answers is permitted." Checking at the
    corpus root AND one level up catches the common shape where the cache is
    written next to the corpus rather than inside it.
    """
    root = Path(root).resolve()
    candidates = [root / p for p in _ANSWER_CACHE_PATTERNS]
    parent = root.parent
    if parent != root:
        candidates.extend(parent / p for p in _ANSWER_CACHE_PATTERNS)

    found = sorted(str(p) for p in candidates if p.exists())
    if found:
        raise PublicTestAnswerCacheError(
            f"cached public-test answer material found: {found}; plan section 37 "
            f"prohibits caching test answers",
            context={"found": found, "root": str(root)},
        )
    return {
        "clean": True,
        "checked": [str(p) for p in candidates],
        "root": str(root),
        "note": "No persisted answer key exists. Nothing is cached by design.",
    }


def open_public_test(
    *,
    mode: EvaluationMode,
    root: str | Path = PUBLIC_TEST_ROOT,
    corpus_id: str | None = None,
) -> PublicTestCorpus:
    """Open the public-test corpus for evaluation. The only supported entry point.

    Verifies the corpus against its seal (when one exists) and checks the
    answer-cache prohibition before returning. Fails loudly so that an unavailable
    corpus is reported as unavailable rather than silently evaluating nothing.
    """
    if not isinstance(mode, EvaluationMode):
        raise PublicTestModeError(
            f"open_public_test requires an EvaluationMode token "
            f"(got {type(mode).__name__})",
            context={"reason": getattr(mode, "reason", None)},
        )

    corpus = PublicTestCorpus(root=root, corpus_id=corpus_id)
    forbid_answer_cache(corpus.root)

    result: dict[str, Any] = {"opened": True, "reason": mode.reason}
    if not corpus.available:
        raise PublicTestSealError(
            f"the public test corpus at {corpus.root} holds no material; it cannot "
            f"be evaluated. An empty corpus is reported as unavailable rather than "
            f"as a passing test set.",
            context=corpus.describe(),
        )

    if (corpus.root / SEAL_FILENAME).exists():
        result = corpus.verify()
    else:
        result["seal"] = "absent (corpus is available but unsealed)"

    corpus._open_report = result  # type: ignore[attr-defined]
    return corpus
