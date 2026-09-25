"""Phase 11 geometry guards — the four requirements that were *measured but
unguarded*.

WHY THIS FILE EXISTS
--------------------
Phase 11 graded eleven requirements. Seven of them (#1, #5, #6, #7, #8, #9,
#11) are already guarded by ``tests/unit/test_optical_sar_croma.py``. The other
four were graded PARTIAL for exactly one reason: the property was *measured*
(recorded in ``artifacts/optical_sar/croma_forward.json``) but **no test would
fail if it changed**. A number that only exists in a JSON blob is a claim, not a
contract.

The four unguarded requirements, and the property each one is about:

    #2  encoder depth is 12, and the SAR encoder is HALF of it (s1=6, s2=12)
    #3  the ViT attention head count is 16
    #4  the ViT patch size is 8
    #10 the vendored ``use_croma.py`` is pinned (DEV-1) and must never drift

The guards below **re-derive** each fact from the vendored implementation
itself — by importing the real ``ViT`` and by parsing the source with ``ast`` —
rather than re-asserting a constant copied out of the same file. A guard that
reads its expected value from the artefact it is guarding is a tautology and
proves nothing; ``test_optical_sar_croma.py`` already asserts the *wrapper*
constants, so restating them here would add no protection.

WHAT IS DELIBERATELY *NOT* INSTANTIATED
---------------------------------------
``PretrainedCROMA.__init__`` calls ``torch.load`` on a 777 MB checkpoint
(``croma_forward.json:checkpoint_bytes``). That is ~1.8 GB resident and is
forbidden here. The plain ``ViT`` class is a different thing entirely: it is
~85 M randomly-initialised parameters, built in well under a second, and is the
exact object whose ``num_heads`` / ``patch_size`` / ``depth`` the requirements
name. Requirement #2's asymmetry lives in ``PretrainedCROMA``'s *construction
expressions* (``depth=int(self.encoder_depth / 2)`` vs ``depth=self.encoder_depth``),
so it is proven structurally with ``ast`` — no checkpoint, no forward pass.

Every assertion here is read-only. The vendored file is only ever opened for
reading and hashing.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("einops")

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The vendored third-party file (DEV-1). Pinned by the invariant baseline.
VENDORED_PATH = REPO_ROOT / "specialists" / "optical_sar" / "vendor" / "use_croma.py"

#: Requirement #10 (DEV-1). These two literals are the PIN. They are intentionally
#: hard-coded here — NOT read from the file — so that re-vendoring the file (or
#: any edit to it) breaks this test. The pin lives in ``tests/`` because
#: ``artifacts/`` is gitignored (``.gitignore`` ``artifacts/``) and so a pin there
#: would not be durable.
VENDORED_SHA256 = "a38567beed29eb08108a47cdc97fe98aec50fd4be0bd98a5266bcd18aafb7c5b"
VENDORED_SIZE = 14_556

#: Requirement #3 / #4 — the plain-ViT geometry.
EXPECTED_NUM_HEADS = 16
EXPECTED_PATCH_SIZE = 8

#: Requirement #2 — base model depth, and the asymmetry it implies.
EXPECTED_ENCODER_DEPTH = 12
EXPECTED_S1_DEPTH = 6   # int(encoder_depth / 2)
EXPECTED_S2_DEPTH = 12  # encoder_depth

#: The measured artefact this re-derivation is cross-checked against.
CROMA_FORWARD_ARTIFACT = REPO_ROOT / "artifacts" / "optical_sar" / "croma_forward.json"


# ---------------------------------------------------------------------------
# Loading the vendored implementation — parameterised by path so the mutation
# proofs can run against a scratch COPY without ever touching the real file.
# ---------------------------------------------------------------------------


def load_vit_class(source_path: Path):
    """Import ``use_croma.py`` from *source_path* and return its ``ViT`` class.

    The module is loaded under a unique name derived from the path so repeated
    loads (the real file and a scratch copy) never collide in ``sys.modules``.
    Loading the module imports ``torch``/``einops`` but constructs nothing and
    loads no weights.
    """
    source_path = Path(source_path)
    module_name = f"_vendored_use_croma_{abs(hash(source_path.as_posix()))}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"could not build an import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ViT


def instantiate_plain_vit(source_path: Path):
    """The base-model ``ViT``: dim=768, depth=12, in_channels=12 (optical)."""
    vit_cls = load_vit_class(source_path)
    return vit_cls(dim=768, depth=EXPECTED_ENCODER_DEPTH, in_channels=12)


def transformer_depth(vit) -> int:
    """The number of transformer layers the ViT ACTUALLY built.

    ``vit.depth`` is an attribute the ViT sets from its ``depth`` argument; on
    its own it is self-referential. ``vit.transformer`` is a ``BaseTransformer``
    whose ``self.layers`` is a ``ModuleList`` with one entry per ``depth``
    (``use_croma.py:249-254``), so this is the real, downstream count.
    """
    return len(vit.transformer.layers)


def attention_head_counts(vit) -> list[int]:
    """The head count that ACTUALLY reaches attention, one per layer.

    ``vit.num_heads`` is what the ViT *declares* (``use_croma.py:306``). The
    heads that run come from the ``BaseTransformer(num_heads=self.num_heads)``
    wiring at ``:311-314``. Both defaults *below* that wiring are 8
    (``BaseTransformer.__init__`` ``:238-246`` and ``Attention.__init__``
    ``:164-168``), so a wiring regression that dropped ``num_heads=self.num_heads``
    would leave ``vit.num_heads == 16`` while every attention layer ran with 8.
    Each ``layers[i]`` is a ``ModuleList([Attention, FFN])``, so ``layers[i][0]``
    is the attention module carrying the live head count.
    """
    return [layer[0].num_heads for layer in vit.transformer.layers]


# ---------------------------------------------------------------------------
# Structural parsing — re-derive the PretrainedCROMA construction facts
# ---------------------------------------------------------------------------


def _parse(source_path: Path) -> ast.Module:
    return ast.parse(Path(source_path).read_text(encoding="utf-8"))


def _is_size_is_base(test: ast.expr) -> bool:
    """True for the ``size == 'base'`` comparison that selects the base branch."""
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "size"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "base"
    )


def _base_branch(source_path: Path) -> ast.If:
    """The ``if size == 'base':`` statement inside ``PretrainedCROMA.__init__``."""
    for node in ast.walk(_parse(source_path)):
        if isinstance(node, ast.If) and _is_size_is_base(node.test):
            return node
    raise AssertionError(f"no `if size == 'base':` branch found in {source_path}")


def _assigned_literal(branch: ast.If, attr: str):
    """The value node of ``self.<attr> = <literal>`` inside *branch*'s body."""
    for stmt in branch.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Attribute)
            and stmt.targets[0].attr == attr
        ):
            return stmt.value
    raise AssertionError(f"`self.{attr} = ...` not found in the base branch")


def base_branch_geometry(source_path: Path) -> dict[str, int]:
    """The four literals the base branch assigns: dim / depth / heads / patch."""
    branch = _base_branch(source_path)
    out: dict[str, int] = {}
    for attr in ("encoder_dim", "encoder_depth", "num_heads", "patch_size"):
        value = _assigned_literal(branch, attr)
        if not isinstance(value, ast.Constant) or not isinstance(value.value, int):
            raise AssertionError(
                f"base branch assigns a non-literal to self.{attr}: {ast.dump(value)}"
            )
        out[attr] = value.value
    return out


def _vit_encoder_call(source_path: Path, attr: str) -> ast.Call:
    """The ``ViT(...)`` call assigned to ``self.<attr>`` (s1_encoder / s2_encoder)."""
    for node in ast.walk(_parse(source_path)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == attr
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "ViT"
        ):
            return node.value
    raise AssertionError(f"`self.{attr} = ViT(...)` not found in {source_path}")


def _depth_keyword(call: ast.Call) -> ast.expr:
    for kw in call.keywords:
        if kw.arg == "depth":
            return kw.value
    raise AssertionError("the ViT(...) call has no `depth=` keyword")


def _depth_descriptor(node: ast.expr) -> tuple:
    """Classify a ``depth=`` expression.

    Returns one of:
        ("full",)        -- ``self.encoder_depth``                 (optical: s2)
        ("int_div", n)   -- ``int(self.encoder_depth / n)``        (SAR: s1, n=2)
        ("other", dump)  -- anything else; the guard then fails
    """
    if isinstance(node, ast.Attribute) and node.attr == "encoder_depth":
        return ("full",)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and len(node.args) == 1
    ):
        inner = node.args[0]
        if (
            isinstance(inner, ast.BinOp)
            and isinstance(inner.op, ast.Div)
            and isinstance(inner.left, ast.Attribute)
            and inner.left.attr == "encoder_depth"
            and isinstance(inner.right, ast.Constant)
            and isinstance(inner.right.value, (int, float))
        ):
            return ("int_div", inner.right.value)
    return ("other", ast.dump(node))


def _resolve_depth(declared: int, descriptor: tuple) -> int:
    if descriptor[0] == "full":
        return declared
    if descriptor[0] == "int_div":
        return int(declared / descriptor[1])
    raise AssertionError(f"unrecognised depth expression: {descriptor}")


def encoder_depths(source_path: Path) -> tuple[int, int, tuple, tuple]:
    """Re-derive ``(s1_depth, s2_depth, s1_descriptor, s2_descriptor)``.

    Reads the declared base depth from the source, reads how each encoder is
    constructed, and *computes* the two depths. Nothing is copied from the
    measured artefact.
    """
    declared = base_branch_geometry(source_path)["encoder_depth"]
    s1_desc = _depth_descriptor(_depth_keyword(_vit_encoder_call(source_path, "s1_encoder")))
    s2_desc = _depth_descriptor(_depth_keyword(_vit_encoder_call(source_path, "s2_encoder")))
    return _resolve_depth(declared, s1_desc), _resolve_depth(declared, s2_desc), s1_desc, s2_desc


def vendored_pin(source_path: Path) -> tuple[int, str]:
    """``(byte_size, sha256)`` of the file on disk, computed from its bytes."""
    raw = Path(source_path).read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


# ===========================================================================
# Requirement #3 / #4 — heads = 16, patch_size = 8
# ===========================================================================


def test_plain_vit_attention_heads_is_16() -> None:
    """#3. 16 heads must reach ATTENTION, not merely be declared on the ViT.

    ``vit.num_heads`` is the DECLARED value (``use_croma.py:306``). The heads
    that actually run come from the ``BaseTransformer(num_heads=self.num_heads)``
    wiring at ``:311-314``, and both defaults below that wiring are 8. So a
    wiring regression that dropped ``num_heads=self.num_heads`` would leave
    ``vit.num_heads == 16`` while every attention layer silently ran with 8 — a
    geometry regression in requirement #3 that reading the attribute alone
    cannot detect. Asserting the per-layer head count is what catches it.
    """
    vit = instantiate_plain_vit(VENDORED_PATH)

    # The declared value...
    assert vit.num_heads == EXPECTED_NUM_HEADS

    # ...and the value that actually reaches attention, for EVERY layer.
    per_layer = attention_head_counts(vit)
    assert per_layer, "the ViT built no attention layers"
    assert set(per_layer) == {EXPECTED_NUM_HEADS}, (
        f"attention layers run with {sorted(set(per_layer))} heads, expected "
        f"{{{EXPECTED_NUM_HEADS}}}; the BaseTransformer/Attention defaults are 8, "
        f"so this means the num_heads wiring was lost"
    )


def test_plain_vit_patch_size_is_8() -> None:
    """#4. The ViT hard-codes an 8x8 patch for both base and large models."""
    vit = instantiate_plain_vit(VENDORED_PATH)
    assert vit.patch_size == EXPECTED_PATCH_SIZE


def test_plain_vit_depth_builds_that_many_transformer_layers() -> None:
    """Depth is WIRED, not hard-coded — build with a NON-default depth.

    Asserting ``vit.depth`` against the same value passed in is self-referential
    and proves nothing. The load-bearing fact is that the transformer really has
    that many layers, so this builds with ``depth=7`` (deliberately NOT the
    model's 12) and asserts the layer count follows. If ``depth`` were ever
    hard-coded or dropped from the ``BaseTransformer`` wiring, this fails.
    """
    vit_cls = load_vit_class(VENDORED_PATH)
    vit = vit_cls(dim=768, depth=7, in_channels=12)

    assert vit.depth == 7
    assert transformer_depth(vit) == 7
    assert len(attention_head_counts(vit)) == 7

    # And the default build really is 12 layers — the two are distinguished.
    assert transformer_depth(instantiate_plain_vit(VENDORED_PATH)) == EXPECTED_ENCODER_DEPTH


def test_base_branch_declares_16_heads_and_patch_size_8() -> None:
    """#3 / #4 re-derived from the ``PretrainedCROMA`` base branch, not the ViT.

    The ViT class and the ``PretrainedCROMA`` base branch are two independent
    places the same two numbers live. Asserting both means a change to either
    is caught.
    """
    geometry = base_branch_geometry(VENDORED_PATH)
    assert geometry["num_heads"] == EXPECTED_NUM_HEADS
    assert geometry["patch_size"] == EXPECTED_PATCH_SIZE


# ===========================================================================
# Requirement #2 — encoder depth 12, with the s1=6 / s2=12 asymmetry
# ===========================================================================


def test_base_branch_declares_encoder_depth_12() -> None:
    geometry = base_branch_geometry(VENDORED_PATH)
    assert geometry["encoder_depth"] == EXPECTED_ENCODER_DEPTH


def test_sar_encoder_is_half_depth_and_optical_is_full_depth() -> None:
    """The asymmetry, asserted as a STRUCTURE rather than as two numbers.

    ``self.s1_encoder`` must be built with ``int(self.encoder_depth / 2)`` and
    ``self.s2_encoder`` with ``self.encoder_depth``. Checking the expressions —
    not merely their current values — is what makes the guard survive a change
    to ``encoder_depth`` itself.
    """
    _s1, _s2, s1_desc, s2_desc = encoder_depths(VENDORED_PATH)
    assert s1_desc == ("int_div", 2), f"SAR depth is not encoder_depth/2: {s1_desc}"
    assert s2_desc == ("full",), f"optical depth is not encoder_depth: {s2_desc}"


def test_recovered_depths_are_6_and_12() -> None:
    """Re-derive (s1, s2) from the source and assert the measured pair."""
    s1, s2, _d1, _d2 = encoder_depths(VENDORED_PATH)
    assert (s1, s2) == (EXPECTED_S1_DEPTH, EXPECTED_S2_DEPTH)


@pytest.mark.skipif(
    not CROMA_FORWARD_ARTIFACT.exists(),
    reason="artifacts/ is gitignored; the Phase 11 forward artefact is absent",
)
def test_recovered_depths_match_the_croma_forward_artifact() -> None:
    """The re-derived pair must equal what the real forward pass measured.

    This closes the loop between the *code* (which the AST guard reads) and the
    *measurement* (which the artefact recorded). ``croma_forward.json`` lives
    under gitignored ``artifacts/``, hence the skip when it is absent — but when
    it is present, the two must agree exactly.
    """
    payload = json.loads(CROMA_FORWARD_ARTIFACT.read_text(encoding="utf-8"))
    block = payload["encoder_depth"]

    s1, s2, _d1, _d2 = encoder_depths(VENDORED_PATH)

    assert block["declared_encoder_depth"] == EXPECTED_ENCODER_DEPTH
    assert block["s1_depth_measured"] == s1
    assert block["s2_depth_measured"] == s2
    assert block["asymmetric"] is (s1 != s2)
    assert (s1, s2) == (EXPECTED_S1_DEPTH, EXPECTED_S2_DEPTH)


# ===========================================================================
# Requirement #10 — the vendored file is pinned (DEV-1)
# ===========================================================================


def test_vendored_use_croma_is_pinned_by_size_and_sha256() -> None:
    """#10. A drift guard: the vendored file must never be silently re-vendored.

    ``use_croma.py`` is third-party code copied into the tree because CROMA is
    not on PyPI (DEV-1). If it is ever replaced or edited, the model's behaviour
    can change with no other symptom. The pin is the only thing standing between
    "vendored at a known revision" and "whatever is on disk".
    """
    size, sha = vendored_pin(VENDORED_PATH)
    assert size == VENDORED_SIZE, f"vendored size drifted: {size} != {VENDORED_SIZE}"
    assert sha == VENDORED_SHA256, f"vendored sha256 drifted: {sha}"


def test_vendored_pin_literals_are_not_read_from_the_file() -> None:
    """Narrow guard: the pin constants are NOT read from the file they pin.

    This test compares two module-level constants against two literals in this
    same file, so it can catch exactly ONE failure mode: someone "fixing" a
    re-vendor by copying the new on-disk hash into ``VENDORED_SHA256`` (and the
    new size into ``VENDORED_SIZE``), which would make the pin self-referential
    and stop it protecting anything.

    It does NOT guard the vendored file itself — that is
    ``test_vendored_use_croma_is_pinned_by_size_and_sha256`` above, which reads
    the file from disk. This test says nothing about the file's contents.
    """
    assert VENDORED_SHA256 == (
        "a38567beed29eb08108a47cdc97fe98aec50fd4be0bd98a5266bcd18aafb7c5b"
    )
    assert VENDORED_SIZE == 14556
