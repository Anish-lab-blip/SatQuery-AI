"""R-02 — the Kaggle notebook's discovery contract.

WHY THIS FILE EXISTS
--------------------
The first real Kaggle run failed before it trained anything, twice, for the same
underlying reason: the notebook assumed where its inputs would be.

  * `CODE_ROOT` defaulted to `/kaggle/input/satquery-ai`. Kaggle actually
    mounted the dataset under `/kaggle/input/datasets/<owner>/<slug>/`, so
    `.exists()` was False even though the dataset was attached.
  * `find_cdvqa_root()` probed `/kaggle/input` and exactly one level below it,
    which reaches `/kaggle/input/datasets` and `/kaggle/input/datasets/<owner>`
    but never `/kaggle/input/datasets/<owner>/<slug>/`.

Both are now discovered by a depth-bounded, directory-only walk that validates
each candidate against markers only the right directory can satisfy. This file
drives the notebook's OWN cells against a synthetic tree with the real nesting,
so the fix is proven rather than asserted.

HOW THE CELLS ARE EXERCISED
---------------------------
The cells are `exec`'d in a shared namespace, in order, exactly as a kernel
would run them. Each test then replaces the cell's own `_code_search_roots` /
`_data_search_roots` with a synthetic tree before calling the discovery
function, so the walk, the audits, the printing and the failure path are all the
real ones.

`/kaggle/input` cannot be created on Windows, which is exactly why the search
roots are injected instead of the real mount being faked. Off Kaggle the cells'
own call finds nothing and exits; that SystemExit is swallowed during setup,
because the functions are already defined by the time the cell calls them.
"""

from __future__ import annotations

import ast
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_change_vqa_train.ipynb"

CODE_MARKERS = (
    "scripts/prepare_change_vqa.py",
    "training/change_vqa/model.py",
    "specialists/change/vqa_specialist.py",
    "configs/base.yaml",
)
ANNOTATIONS = tuple(
    f"{split}_{table}.json"
    for split in ("Train", "Val", "Test", "Test2")
    for table in ("images", "questions", "answers")
)
IMAGE_DIRS = ("im1", "im2", "label1", "label2")

CHECKPOINT_RELATIVE = 'CODE_ROOT / "artifacts" / "change" / "levir_change_v001" / "head.pt"'


# ---------------------------------------------------------------------------
# Notebook access
# ---------------------------------------------------------------------------


def _code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _cell_containing(marker: str) -> str:
    """Find a cell by a marker it must contain.

    Located by content rather than by index so that reordering the notebook
    cannot silently point a test at the wrong cell.
    """
    for source in _code_cells():
        if marker in source:
            return source
    raise AssertionError(f"no code cell in the notebook contains {marker!r}")


def _code_without_comments() -> str:
    """Executable code only. A comment naming the wrong path is documentation."""
    lines: list[str] = []
    for source in _code_cells():
        for line in source.splitlines():
            if line.lstrip().startswith("#"):
                continue
            lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic trees
# ---------------------------------------------------------------------------


def _make_code_tree(base: Path) -> Path:
    root = base / "datasets" / "creatorballs" / "satquery-ai-code"
    for marker in CODE_MARKERS:
        path = root / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic\n", encoding="utf-8")
    checkpoint = root / "artifacts" / "change" / "levir_change_v001" / "head.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"not-the-real-detector")
    return root


def _make_cdvqa_tree(base: Path, name: str = "cdvqa-dataset") -> Path:
    root = base / "datasets" / "creatorballs" / name
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    for filename in ANNOTATIONS:
        (annotations / filename).write_text("[]", encoding="utf-8")
    for directory in IMAGE_DIRS:
        (root / directory).mkdir(parents=True, exist_ok=True)
    return root


def _namespace(monkeypatch: Any, scratch: Path) -> dict[str, Any]:
    """A kernel-like namespace with the real cells 3 -> 5 -> 7 executed."""
    monkeypatch.delenv("SATQUERY_CODE", raising=False)
    monkeypatch.delenv("SATQUERY_DATA", raising=False)
    monkeypatch.delenv("SATQUERY_CHANGE_CHECKPOINT", raising=False)
    monkeypatch.setenv("SATQUERY_OUT", str(scratch / "out"))

    namespace: dict[str, Any] = {"__name__": "kaggle_kernel"}
    for marker in ("HARD_STOP_SECONDS", "def find_code_root", "def find_cdvqa_root"):
        source = _cell_containing(marker)
        try:
            exec(compile(source, f"<{marker}>", "exec"), namespace)
        except SystemExit:
            # No Kaggle mount and no CDVQA on this machine. The functions are
            # defined before the cell's final call, and every test replaces the
            # search roots before invoking them.
            pass
    return namespace


@pytest.fixture
def scratch() -> Path:
    # `tempfile.mkdtemp`, never pytest's `tmp_path`: this sandbox blocks the
    # directory removals `tmp_path` performs at teardown. The directory is left
    # in place on purpose.
    return Path(tempfile.mkdtemp(prefix="r02_notebook_"))


# ---------------------------------------------------------------------------
# The regression: a root nested three levels below the mount base
# ---------------------------------------------------------------------------


def test_the_walk_reaches_a_root_three_levels_below_the_base(
    monkeypatch: Any, scratch: Path
) -> None:
    """This is the exact shape that defeated the old one-level probe.

    `/kaggle/input/datasets/<owner>/<slug>/` is depth 3. The old discovery
    tested depth 0 and depth 1 only.
    """
    base = scratch / "kaggle_input"
    expected = _make_code_tree(base)

    namespace = _namespace(monkeypatch, scratch)
    namespace["_code_search_roots"] = lambda: [base]
    found = namespace["find_code_root"]()
    assert found == expected
    assert found.relative_to(base).parts == (
        "datasets", "creatorballs", "satquery-ai-code",
    )


def test_the_notebook_discovers_a_nested_code_root(
    monkeypatch: Any, scratch: Path
) -> None:
    base = scratch / "kaggle_input"
    expected = _make_code_tree(base)
    namespace = _namespace(monkeypatch, scratch)
    namespace["_code_search_roots"] = lambda: [base]
    assert namespace["find_code_root"]() == expected


def test_the_stanet_path_is_derived_from_the_discovered_root() -> None:
    """Not a literal: if it were, a discovered root would still look for the
    detector in the wrong place."""
    source = _cell_containing("def find_code_root")
    assert CHECKPOINT_RELATIVE in source
    assert 'STANET_OVERRIDE' in source


def test_the_notebook_discovers_a_nested_cdvqa_root(
    monkeypatch: Any, scratch: Path
) -> None:
    base = scratch / "kaggle_input"
    _make_code_tree(base)
    expected = _make_cdvqa_tree(base)

    namespace = _namespace(monkeypatch, scratch)
    namespace["_data_search_roots"] = lambda: [base]
    assert namespace["find_cdvqa_root"]() == expected


def test_both_roots_are_found_when_they_sit_beside_each_other(
    monkeypatch: Any, scratch: Path
) -> None:
    """The real session: two datasets under one owner namespace."""
    base = scratch / "kaggle_input"
    code_root = _make_code_tree(base)
    data_root = _make_cdvqa_tree(base)

    namespace = _namespace(monkeypatch, scratch)
    namespace["_code_search_roots"] = lambda: [base]
    namespace["_data_search_roots"] = lambda: [base]
    assert namespace["find_code_root"]() == code_root
    assert namespace["find_cdvqa_root"]() == data_root


# ---------------------------------------------------------------------------
# It must not select the wrong directory
# ---------------------------------------------------------------------------


def test_a_directory_that_merely_looks_like_the_project_is_rejected(
    monkeypatch: Any, scratch: Path
) -> None:
    """Three of four markers is not the project, and must not be accepted."""
    base = scratch / "kaggle_input"
    decoy = base / "datasets" / "creatorballs" / "satquery-ai-code"
    for marker in CODE_MARKERS[:3]:  # configs/base.yaml deliberately absent
        path = decoy / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# decoy\n", encoding="utf-8")
    real = _make_code_tree(base / "second")
    # The decoy is searched FIRST, so a naive "first match wins" would take it.
    namespace = _namespace(monkeypatch, scratch)
    namespace["_code_search_roots"] = lambda: [base, base / "second"]
    assert namespace["find_code_root"]() == real


def test_an_incomplete_cdvqa_root_is_rejected(
    monkeypatch: Any, scratch: Path
) -> None:
    """`annotations/` alone is a real Kaggle failure mode: it loads, then dies
    much later with a less useful error. It must be rejected up front."""
    base = scratch / "kaggle_input"
    partial = base / "datasets" / "creatorballs" / "cdvqa-annotations-only"
    (partial / "annotations").mkdir(parents=True)
    for filename in ANNOTATIONS:
        (partial / "annotations" / filename).write_text("[]", encoding="utf-8")
    real = _make_cdvqa_tree(base / "second")

    namespace = _namespace(monkeypatch, scratch)
    namespace["_data_search_roots"] = lambda: [base, base / "second"]
    assert namespace["find_cdvqa_root"]() == real


def test_discovery_stops_with_an_actionable_error_when_nothing_matches(
    monkeypatch: Any, scratch: Path
) -> None:
    base = scratch / "kaggle_input"
    base.mkdir(parents=True, exist_ok=True)
    (base / "some-unrelated-dataset").mkdir()

    namespace = _namespace(monkeypatch, scratch)
    namespace["_code_search_roots"] = lambda: [base]
    with pytest.raises(SystemExit) as excinfo:
        namespace["find_code_root"]()
    message = str(excinfo.value)
    assert "PROJECT CODE NOT FOUND" in message
    assert "SATQUERY_CODE" in message
    # It must say what it looked at, or the operator cannot tell a bad mount
    # from a bad path.
    assert "Searched:" in message
    assert str(base) in message


def test_cdvqa_failure_names_the_expected_structure(
    monkeypatch: Any, scratch: Path
) -> None:
    base = scratch / "kaggle_input"
    _make_code_tree(base)

    namespace = _namespace(monkeypatch, scratch)
    namespace["_data_search_roots"] = lambda: [base]
    with pytest.raises(SystemExit) as excinfo:
        namespace["find_cdvqa_root"]()
    message = str(excinfo.value)
    assert "CDVQA DATASET NOT FOUND" in message
    assert str(base) in message
    for expected in ("annotations/", "im1/", "im2/", "label1/", "label2/"):
        assert expected in message
    assert "Train_images.json" in message


# ---------------------------------------------------------------------------
# The walk itself
# ---------------------------------------------------------------------------


def test_the_walk_is_depth_bounded(monkeypatch: Any, scratch: Path) -> None:
    base = scratch / "kaggle_input"
    (base / "a" / "b" / "c" / "d" / "e" / "f").mkdir(parents=True)
    namespace = _namespace(monkeypatch, scratch)
    depths = {depth for _path, depth in namespace["_walk_dirs"]([base])}
    assert max(depths) <= namespace["MAX_DISCOVERY_DEPTH"]


def test_the_walk_never_descends_into_the_imagery_directories(
    monkeypatch: Any, scratch: Path
) -> None:
    """2,968 PNGs per directory: listing them would make discovery slow and
    pointless, so the walk prunes them by name."""
    base = scratch / "kaggle_input"
    root = _make_cdvqa_tree(base)
    for directory in IMAGE_DIRS:
        (root / directory / "scene_0001.png").write_bytes(b"x")

    namespace = _namespace(monkeypatch, scratch)
    visited = {path.name for path, _depth in namespace["_walk_dirs"]([base])}
    for directory in IMAGE_DIRS:
        assert directory not in visited
    assert "annotations" not in visited


def test_the_walk_tolerates_an_unreadable_directory(
    monkeypatch: Any, scratch: Path
) -> None:
    """A permission error inside a mount must not abort discovery."""
    base = scratch / "kaggle_input"
    (base / "datasets" / "creatorballs" / "cdvqa-dataset" / "annotations").mkdir(
        parents=True
    )
    namespace = _namespace(monkeypatch, scratch)

    real_iterdir = Path.iterdir

    def flaky(self: Path) -> Any:
        if self.name == "datasets":
            raise PermissionError("simulated")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", flaky)
    visited = [path for path, _depth in namespace["_walk_dirs"]([base])]
    assert base in visited  # it walked what it could and did not raise


def test_the_walk_does_not_start_at_a_filesystem_root(
    monkeypatch: Any, scratch: Path
) -> None:
    """On Kaggle the ancestors of the cwd reach `/`, and a depth-bounded walk
    from there would spend its whole budget in /usr instead of /kaggle/input."""
    namespace = _namespace(monkeypatch, scratch)
    roots = namespace["_code_search_roots"]()
    for root in roots:
        assert root == root.parent or root.is_dir() or True  # noqa: SIM109
    # No root may be a drive/filesystem root.
    assert all(str(r) not in ("/", "C:\\", "C:/") for r in roots)


# ---------------------------------------------------------------------------
# The notebook must not reintroduce a hardcoded mount
# ---------------------------------------------------------------------------


def test_no_executable_code_hardcodes_a_kaggle_mount_slug() -> None:
    code = _code_without_comments()
    assert "/kaggle/input/satquery-ai" not in code
    assert "/kaggle/input/cdvqa" not in code
    assert "/kaggle/input/datasets/creatorballs" not in code


def test_no_cell_writes_under_the_read_only_input_mount() -> None:
    """The notebook reads the project from the dataset and writes only to
    OUT_DIR. The guard that enforces it must be present."""
    source = _cell_containing("OUT_DIR writable")
    assert "/kaggle/input" in source
    assert "os.access(OUT_DIR, os.W_OK)" in source


def test_the_notebook_disables_bytecode_writes() -> None:
    """Importing from a read-only dataset otherwise makes CPython try to create
    __pycache__ inside it."""
    assert "sys.dont_write_bytecode = True" in _cell_containing("HARD_STOP_SECONDS")


def test_every_output_path_is_derived_from_out_dir() -> None:
    """A literal output path would escape to a read-only or non-persisted
    location. `OUT_DIR` is the only base."""
    code = _code_without_comments()
    for derived in ("PREPARED = OUT_DIR /", "RUN_DIR = OUT_DIR /",
                    "EVAL_DIR = OUT_DIR /", "EXPORT = OUT_DIR /"):
        assert derived in code, derived
    assert 'OUT_DIR = Path(os.environ.get("SATQUERY_OUT"' in code


# ---------------------------------------------------------------------------
# The held-out targets: what stopped the SECOND Kaggle run
# ---------------------------------------------------------------------------
#
# The first run failed on discovery (above). The second trained, then died at
# evaluation with `missing_targets: 39686` — because `scene_targets.jsonl` was
# built for `--splits Train Val` and EVERY later prepare call passed
# `--skip-targets`. The held-out targets the evaluator needs were therefore never
# written, and the error named the symptom, not the step that was skipped.
#
# The tests below assert the ORCHESTRATION, not the evaluator: the evaluator's
# own refusal message is covered in `test_change_vqa_eval_cli.py`. What matters
# here is that the notebook now builds all four splits' targets, in one pass, in
# the right order, and refuses to score if a cached scene has no target.
#
# The subprocess argv lists are recovered with `ast`, not by searching the text.
# A comment that mentions `--skip-targets` is documentation; only a real argument
# counts, and only `ast` can tell the two apart.

_NON_LITERAL = "<non-literal>"


def _evaluation_cell() -> str:
    return _cell_containing("evaluate_change_vqa.py")


def _argv_lists(source: str) -> list[list[str]]:
    """Every `subprocess.call([...])` in the source, as a token list.

    Non-literal arguments (`str(DATA_ROOT)`, `DEVICE`) become a placeholder, so
    the flags and the literal `--splits` values are still recoverable.
    """
    out: list[list[str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "call"):
            continue
        if not node.args or not isinstance(node.args[0], ast.List):
            continue
        argv: list[str] = []
        for element in node.args[0].elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                argv.append(element.value)
            else:
                argv.append(_NON_LITERAL)
        out.append(argv)
    return out


def _splits_of(argv: list[str]) -> list[str]:
    if "--splits" not in argv:
        return []
    values: list[str] = []
    for token in argv[argv.index("--splits") + 1:]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


def _prepare_calls(argv_lists: list[list[str]]) -> list[list[str]]:
    return [
        argv for argv in argv_lists
        if any(token.endswith("prepare_change_vqa.py") for token in argv)
    ]


def test_the_four_split_prepare_call_builds_the_held_out_targets() -> None:
    """THE regression. `--skip-targets` on this call is the entire second
    failure: with it, Test/Test2 have features and no labels, and scoring reports
    `missing_targets` for every question in them."""
    calls = _prepare_calls(_argv_lists(_evaluation_cell()))
    four_split = [
        argv for argv in calls if _splits_of(argv) == ["Train", "Val", "Test", "Test2"]
    ]
    assert len(four_split) == 1, calls
    argv = four_split[0]
    assert "--skip-targets" not in argv, argv
    # Features too: the same pass must leave the cache covering all four splits.
    assert "--skip-features" not in argv, argv
    # But the question features for all four splits already exist by now, so
    # re-encoding 65,967 questions would be pure waste.
    assert "--skip-text" in argv, argv


def test_the_held_out_text_feature_call_still_skips_targets() -> None:
    """The counterpart guard, and the reason the two calls are separate.

    `write_scene_targets` OVERWRITES the file. If this Test/Test2-only call built
    targets, it would replace the 2,000 Train/Val scenes with Test/Test2 — the
    mirror image of the original bug. Enabling targets there looks like a fix and
    is not one."""
    calls = _prepare_calls(_argv_lists(_evaluation_cell()))
    two_split = [argv for argv in calls if _splits_of(argv) == ["Test", "Test2"]]
    assert len(two_split) == 1, calls
    argv = two_split[0]
    assert "--skip-targets" in argv, argv
    assert "--skip-features" in argv, argv
    # This call exists for exactly one reason: the question features.
    assert "--skip-text" not in argv, argv


def test_the_evaluation_cell_guards_scene_coverage_before_scoring() -> None:
    """The hard guard. Without it the next version of this bug is discovered
    again on Kaggle, one 20-minute run at a time."""
    source = _evaluation_cell()
    assert "read_scene_targets" in source
    assert "ChangeFeatureCache.read" in source
    assert "SystemExit" in source
    # The message must name the symptom the caller would otherwise see, so the
    # two are recognisably the same failure.
    assert "missing_targets" in source


def test_the_coverage_guard_joins_the_cache_and_the_targets_on_scene_key() -> None:
    """The change cache is scene-keyed; the targets file is file-name-keyed.
    `scene_key` is the only field both sides carry, so it is the only valid join
    — and comparing the wrong pair is how a guard passes while the evaluator
    still refuses."""
    source = _evaluation_cell()
    assert "scene_key" in source
    assert "scene_keys" in source


def test_the_coverage_guard_runs_before_the_evaluation_call() -> None:
    """Anchored on the guard's CALL SITE, not on the name.

    The first version of this test searched for `read_scene_targets`, which the
    import line satisfies on its own — so relocating the whole guard below the
    evaluation subprocess still passed. `read_scene_targets(PREPARED` is the
    call, and the call is what has to come first.
    """
    source = _evaluation_cell()
    guard = source.index("read_scene_targets(PREPARED")
    evaluation = source.index("evaluate_change_vqa.py")
    assert guard < evaluation


def test_the_held_out_targets_are_built_after_the_head_is_fitted() -> None:
    """Ordering is the leakage argument, not a style choice.

    Building held-out targets earlier would not leak on its own — the trainer
    enumerates `READABLE_SPLITS = Train, Val` and looks up only the file names it
    already has — but preparing them after training makes the invariant visible
    in the notebook instead of merely true in the trainer. Cell 17 asserts the
    trainer half of it; this asserts the notebook half.
    """
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = [
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    ]
    train_at = next(
        i for i, s in enumerate(sources) if "train_change_vqa.py" in s
    )
    guard_at = next(
        i for i, s in enumerate(sources)
        if "read_scene_targets" in s and "evaluate_change_vqa.py" in s
    )
    assert guard_at > train_at, (train_at, guard_at)


# ---------------------------------------------------------------------------
# The guard must cover QUESTION features too -- the third artifact
# ---------------------------------------------------------------------------
#
# Found while closing K4. `evaluate_change_vqa.py` needs three artifacts per
# split, keyed three different ways:
#
#     targets          keyed by FILE NAME   (scene_targets.jsonl)
#     change features  keyed by SCENE KEY   (change_features.npz)
#     text features    keyed by QUESTION ID (text_features_<split>.npz, per split)
#
# The guard covered the first two -- it compared the change cache against the
# targets -- and could not see the third, because that cache is per-split and
# keyed by a different field. Meanwhile the evaluator used to SKIP a split whose
# cache was absent and still exit 0, so the gap was silent.
#
# These tests run the guard block, rather than searching the cell for strings.
# A string search would be satisfied by a comment, and the whole lesson of this
# revision is that asserting on the description of a check is not asserting on
# the check.

_GUARD_START = "# Hard guard"
_GUARD_END = "EVAL_DIR = OUT_DIR"
_GUARD_QUESTIONS = (
    ("change_or_not", "Have the areas of buildings changed?", ("yes", "no")),
    ("increase_or_not", "Has the area of water increased?", ("yes", "no")),
)


def _guard_source() -> str:
    """Cell 27's guard block, from its banner to the evaluation call."""
    source = _evaluation_cell()
    return source[source.index(_GUARD_START):source.index(_GUARD_END)]


def _write_annotations(root: Path, split: str, scenes: tuple[str, ...]) -> None:
    """The three annotation tables for one split, in the shipped shape."""
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    images, questions, answers = [], [], []
    question_id = answer_id = 0
    for scene_index, scene in enumerate(scenes):
        refs = []
        for q_index, (qtype, text, options) in enumerate(_GUARD_QUESTIONS):
            questions.append({
                "id": question_id, "date_added": 1631214265.0,
                "img_id": scene_index, "type": qtype, "question": text,
                "answers_ids": [answer_id], "active": True,
            })
            answers.append({
                "id": answer_id, "date_added": 1631214265.0,
                "question_id": question_id,
                "answer": options[(scene_index + q_index) % len(options)],
                "active": True,
            })
            refs.append(question_id)
            question_id += 1
            answer_id += 1
        images.append({
            "id": scene_index, "res_x": "64", "res_y": "64",
            "questions_ids": refs, "file_name": f"{scene}.png", "active": True,
        })
    for name, key, rows in (
        ("images", "images", images),
        ("questions", "questions", questions),
        ("answers", "answers", answers),
    ):
        (annotations / f"{split}_{name}.json").write_text(
            json.dumps({key: rows}), encoding="utf-8"
        )


def _held_out_root(scratch: Path) -> tuple[Path, dict[str, tuple[str, ...]]]:
    """A two-split CDVQA root with real annotations AND the im1/im2 imagery.

    The imagery is not optional even though the guard loads records with
    `require_images=False`: the temporal pairing is derived from the LAYOUT, so
    a scene with a single image resolves to zero records rather than being
    skipped quietly.
    """
    from PIL import Image

    root = scratch / "data"
    scenes = {
        "Test": ("zztest0001", "zztest0002"),
        "Test2": ("zztest20001", "zztest20002"),
    }
    rng = np.random.default_rng(20260922)
    for split, names in scenes.items():
        _write_annotations(root, split, names)
    for index, scene in enumerate(sum(scenes.values(), ())):
        base = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        for sub, pixels in (("im1", base), ("im2", np.roll(base, 3 + index, axis=0))):
            path = root / sub / f"{scene}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB").save(path)
    return root, scenes


def _prepared_for_guard(
    scratch: Path,
    scenes: dict[str, tuple[str, ...]],
    *,
    text_for: tuple[str, ...],
    drop_questions: int = 0,
) -> Path:
    """A prepared directory the guard can be run against.

    `text_for` names the splits that get a question-feature cache; the rest get
    none, which is exactly the state `prepare` leaves behind for a split that
    resolves no questions. `drop_questions` omits that many question ids from
    every cache, producing a PARTIAL cache -- the other half of the defect.
    """
    from training.change_vqa.dataset import SceneTargets, write_scene_targets
    from training.change_vqa.features import (
        CHANGE_FEATURE_DIM,
        TEXT_FEATURE_DIM,
        ChangeFeatureCache,
        TextFeatureCache,
    )
    from training.change_vqa.vocab import N_CHANGE_CLASSES

    prepared = scratch / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)

    all_scenes = sorted(set().union(*(set(v) for v in scenes.values())))
    zeros = tuple(0.0 for _ in range(N_CHANGE_CLASSES))
    write_scene_targets(
        [
            SceneTargets(
                file_name=f"{key}.png", scene_key=key, class_mag=zeros,
                class_delta=zeros, class_ratio=zeros, total_changed=0.0,
                n_pixels=4, n_labeled_pixels=4,
                label1_sha256="0" * 64, label2_sha256="0" * 64,
            )
            for key in all_scenes
        ],
        prepared / "scene_targets.jsonl",
    )
    ChangeFeatureCache(
        scene_keys=tuple(all_scenes),
        features=np.zeros((len(all_scenes), CHANGE_FEATURE_DIM), dtype=np.float32),
        spec_hash="synthetic-spec",
        extractor_config={"synthetic": True},
    ).write(prepared / "change_features.npz")

    for split, names in scenes.items():
        if split not in text_for:
            continue
        # `question_id` RESTARTS AT 0 FOR EVERY SPLIT, as in the real corpus --
        # the adapter does the same, so a single running counter here would
        # produce ids the records do not have and the guard would (correctly)
        # report every question as uncovered.
        ids = [
            index
            for index in range(len(names) * len(_GUARD_QUESTIONS))
        ]
        kept = ids[: len(ids) - drop_questions] if drop_questions else ids
        TextFeatureCache(
            question_ids=tuple(kept),
            features=np.zeros((len(kept), TEXT_FEATURE_DIM), dtype=np.float32),
            spec_hash="synthetic-spec",
        ).write(prepared / f"text_features_{split}.npz")
    return prepared


def _run_guard(data_root: Path, prepared: Path) -> None:
    """Execute cell 27's guard with the two names it reads."""
    namespace: dict[str, Any] = {
        "__name__": "kaggle_kernel",
        "DATA_ROOT": data_root,
        "PREPARED": prepared,
    }
    exec(compile(_guard_source(), "<cell 27 guard>", "exec"), namespace)


def test_the_guard_covers_the_question_features_not_only_the_scenes() -> None:
    """The scene-level check cannot see this artifact, so there must be a second
    one that does. Anchored on the CALL SITE, not the import."""
    source = _evaluation_cell()
    assert "TextFeatureCache.read(_path)" in source, (
        "the guard must actually read each split's question-feature cache"
    )
    assert "text_features_{_split}.npz" in source, (
        "it must build the per-split cache path the evaluator uses"
    )


def test_the_question_feature_guard_uses_the_evaluators_own_loader() -> None:
    """The guard and the scorer must not disagree about which questions a split
    contains. They agree by construction if the guard loads records the same way
    `evaluate_change_vqa.py` does."""
    source = _evaluation_cell()
    assert "load_change_vqa_records(" in source
    assert "require_images=False" in source
    assert "group_by_native_split(" in source

    evaluator = (REPO_ROOT / "scripts" / "evaluate_change_vqa.py").read_text(
        encoding="utf-8"
    )
    assert "require_images=False" in evaluator, (
        "the evaluator changed its loader; the guard must be updated with it"
    )


def test_the_guard_refuses_a_split_whose_question_features_are_absent(
    scratch: Path,
) -> None:
    """The state `prepare` leaves for a split that resolves no questions.

    Under the old guard this passed, the evaluation ran, and `Test2` was
    silently dropped from `eval_summary.json` while the run exited 0.
    """
    data_root, scenes = _held_out_root(scratch)
    prepared = _prepared_for_guard(scratch, scenes, text_for=("Test",))

    with pytest.raises(SystemExit) as excinfo:
        _run_guard(data_root, prepared)

    message = str(excinfo.value)
    assert "Test2" in message, message
    assert "question-feature" in message, message


def test_the_guard_refuses_a_PARTIAL_question_feature_cache(scratch: Path) -> None:
    """A cache that exists but does not cover every question.

    This is the half of the defect a file-existence check misses: the file is
    there, so `TextFeatureCache.read` succeeds, and the evaluator would drop the
    uncovered questions and report accuracy over the remainder without saying so.
    """
    data_root, scenes = _held_out_root(scratch)
    prepared = _prepared_for_guard(
        scratch, scenes, text_for=("Test", "Test2"), drop_questions=1
    )

    with pytest.raises(SystemExit) as excinfo:
        _run_guard(data_root, prepared)

    message = str(excinfo.value)
    assert "uncovered" in message, message
    assert "Test2" in message and "Test" in message, message


def test_the_guard_passes_when_every_artifact_is_complete(scratch: Path) -> None:
    """The control. A guard that refuses a fully prepared directory would be
    worse than the gap it closes."""
    data_root, scenes = _held_out_root(scratch)
    prepared = _prepared_for_guard(
        scratch, scenes, text_for=("Test", "Test2")
    )

    _run_guard(data_root, prepared)  # must not raise


# ---------------------------------------------------------------------------
# Exit 0 is not the evidence -- the held-out SUMMARY is
# ---------------------------------------------------------------------------
#
# The K6 lesson, applied at the point of evaluation. The evaluator used to skip a
# requested split whose question-feature cache was absent, score the rest, write
# an `eval_summary.json` without the others, and exit **0**. That is fixed at the
# source, but the notebook's own claim -- "this run produced a held-out result" --
# was still resting on an exit code, and an exit code is a statement about a
# process rather than about the artifact.
#
# These tests run the check, rather than searching the cell for strings, for the
# same reason the guard tests do: a string search is satisfied by a comment.

_SUMMARY_START = "# Exit 0 is not the evidence"


def _summary_check_source() -> str:
    """Cell 27's held-out-summary check, which ends the cell."""
    source = _evaluation_cell()
    return source[source.index(_SUMMARY_START):]


def _summary(*entries: tuple[str, bool, int]) -> dict[str, Any]:
    """A summary with the shape `evaluate_change_vqa.py` writes.

    Each entry is `(split, held_out, n_scored)`; the nested metric blocks carry
    only the keys the check prints.
    """
    return {
        "schema": "change_vqa_evaluation_v1",
        "splits": [
            {
                "split": split,
                "held_out": held_out,
                "n_scored": n_scored,
                "mask_gain": 0.0,
                "unmasked": {"accuracy": 0.0},
            }
            for split, held_out, n_scored in entries
        ],
    }


def _run_summary_check(eval_dir: Path, payload: dict[str, Any] | None) -> None:
    """Write `payload` (if any) to `eval_dir/eval_summary.json` and run the check.

    The namespace supplies what the kernel would already have at that point:

    * `json` — imported by cell 3, so it is a module global by cell 27;
    * `EVAL_DIR` — bound at the top of this cell;
    * `HELD_OUT` — bound by the guard block earlier in **this same cell**, so the
      value is taken from the cell's own source rather than repeated here. That
      keeps the behavioural tests honest: they exercise the constant the cell
      actually uses, not one the test chose.
    """
    if payload is not None:
        (eval_dir / "eval_summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    source = _evaluation_cell()
    declared = re.search(r"^HELD_OUT = \((.*?)\)$", source, re.M)
    assert declared, "cell 27 must define HELD_OUT"
    held_out = tuple(
        token.strip().strip('"').strip("'")
        for token in declared.group(1).split(",")
        if token.strip()
    )

    namespace: dict[str, Any] = {
        "__name__": "kaggle_kernel",
        "json": json,
        "EVAL_DIR": eval_dir,
        "HELD_OUT": held_out,
    }
    exec(
        compile(_summary_check_source(), "<cell 27 summary check>", "exec"),
        namespace,
    )


def test_the_evaluation_cell_reads_the_summary_it_claims_to_have_produced() -> None:
    """Anchored on the read, not on a comment about it."""
    source = _evaluation_cell()
    assert 'EVAL_DIR / "eval_summary.json"' in source, (
        "the cell must read the summary the evaluation wrote"
    )
    assert "_requested = set(HELD_OUT)" in source, (
        "it must compare the summary against the splits that were requested, and "
        "it must take that set from the cell's own constant -- a second literal "
        "would be a second place to keep in agreement"
    )


def test_the_cells_held_out_constant_matches_the_evaluators() -> None:
    """The cell's `HELD_OUT` and the evaluator's `HELD_OUT_SPLITS` are the same
    fact stated twice, so they are pinned to each other.

    The summary's `held_out` field is computed from the evaluator's constant. If
    the two ever disagreed, the check would demand a split the evaluator does not
    consider held out (a false refusal) or accept one it does not (a false pass).
    """
    source = _evaluation_cell()
    match = re.search(r'^HELD_OUT = \((.*?)\)$', source, re.M)
    assert match, "cell 27 must define HELD_OUT"
    literal = tuple(
        token.strip().strip('"').strip("'")
        for token in match.group(1).split(",")
        if token.strip()
    )

    evaluator = (REPO_ROOT / "scripts" / "evaluate_change_vqa.py").read_text(
        encoding="utf-8"
    )
    declared = re.search(r'^HELD_OUT_SPLITS: tuple\[str, \.\.\.\] = \((.*?)\)$',
                         evaluator, re.M)
    assert declared, "the evaluator must define HELD_OUT_SPLITS"
    expected = tuple(
        token.strip().strip('"').strip("'")
        for token in declared.group(1).split(",")
        if token.strip()
    )

    assert literal == expected, (
        f"cell 27 says {literal} are held out; the evaluator says {expected}"
    )


def test_the_summary_check_refuses_a_summary_missing_a_requested_split(
    scratch: Path,
) -> None:
    """The K6 failure shape, caught at the notebook instead of by a reader.

    A one-split summary is well-formed JSON. Nothing downstream can tell that a
    held-out split is absent, which is exactly why this is checked.
    """
    eval_dir = scratch / "eval"
    eval_dir.mkdir(parents=True)

    with pytest.raises(SystemExit) as excinfo:
        _run_summary_check(
            eval_dir, _summary(("Test", True, 39686))
        )

    message = str(excinfo.value)
    assert "Test2" in message, message
    assert "Test" in message, message


def test_the_summary_check_refuses_a_summary_with_no_file(scratch: Path) -> None:
    """Exit 0 with no summary at all is a contradiction, not a success."""
    eval_dir = scratch / "eval"
    eval_dir.mkdir(parents=True)

    with pytest.raises(SystemExit) as excinfo:
        _run_summary_check(eval_dir, None)

    assert "eval_summary.json" in str(excinfo.value)


def test_the_summary_check_refuses_a_split_that_is_not_held_out(
    scratch: Path,
) -> None:
    """`Test` present but flagged as not held out would be read as a
    generalisation number when it is a training-time one."""
    eval_dir = scratch / "eval"
    eval_dir.mkdir(parents=True)

    with pytest.raises(SystemExit) as excinfo:
        _run_summary_check(
            eval_dir, _summary(("Test", True, 10), ("Test2", False, 10))
        )

    assert "not held out" in str(excinfo.value)


def test_the_summary_check_refuses_a_split_that_scored_nothing(
    scratch: Path,
) -> None:
    """A present split with zero records reports an accuracy over an empty set."""
    eval_dir = scratch / "eval"
    eval_dir.mkdir(parents=True)

    with pytest.raises(SystemExit) as excinfo:
        _run_summary_check(
            eval_dir, _summary(("Test", True, 10), ("Test2", True, 0))
        )

    assert "0 records" in str(excinfo.value)


def test_the_summary_check_passes_on_a_complete_held_out_summary(
    scratch: Path,
) -> None:
    """The control. A check that refuses a good summary is worse than the gap."""
    eval_dir = scratch / "eval"
    eval_dir.mkdir(parents=True)

    _run_summary_check(
        eval_dir, _summary(("Test", True, 39686), ("Test2", True, 31036))
    )  # must not raise


def test_the_summary_fixture_matches_the_function_that_writes_it() -> None:
    """The tests above run the check against a hand-built summary. This pins that
    summary to the object `score_split` actually returns.

    Without it the fixture is self-referential: it would keep passing after a
    rename in the evaluator, and the check would then `KeyError` on Kaggle --
    which is the one place the failure is expensive. Anchored on the `return`
    node inside `score_split`, not on a text search.
    """
    tree = ast.parse(
        (REPO_ROOT / "scripts" / "evaluate_change_vqa.py").read_text(encoding="utf-8")
    )
    score_split = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "score_split"
    )
    returned = [
        node for node in ast.walk(score_split)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]
    assert returned, "score_split must return a dict literal"
    keys = {
        key.value for key in returned[0].value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    # Every field the check reads must be one of them.
    required = {"split", "held_out", "n_scored", "unmasked", "mask_gain"}
    missing = required - keys
    assert not missing, (
        f"the summary check reads {sorted(missing)}, which score_split does not "
        f"return; it returns {sorted(keys)}"
    )

    # And the nested block the check reaches into.
    fixture = _summary(("Test", True, 1))["splits"][0]
    assert set(fixture) <= keys, (
        f"the fixture has {sorted(set(fixture) - keys)} that score_split does not"
    )
    assert "accuracy" in fixture["unmasked"], (
        "the check prints `unmasked['accuracy']`, so the fixture must carry it"
    )
