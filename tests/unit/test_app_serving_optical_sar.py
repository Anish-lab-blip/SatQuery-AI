"""Regression guards for the optical-SAR serving wiring (Phase 14 / Pass 16).

WHAT THESE GUARD, AND WHY
-------------------------
`specialists/optical_sar/specialist.py` builds CROMA only when it is handed a
`checkpoint_path` that EXISTS:

    if checkpoint_path is not None and Path(checkpoint_path).exists():

`croma.checkpoint_path` is absent from `configs/base.yaml`, and must stay absent
-- `Config.hash` is a sha256 over the whole registry with no exclusion
mechanism, the shipped artifacts record `78f1e3700da15aa1`, and
`scripts/eval_change.py` refuses to score on drift (exit 3). So before this
wiring the default serving composition passed no path, that gate was False, and
`optical_sar` came back DEGRADED ("no encoder; running on fallback") while the
verified checkpoint sat in the Hub cache the whole time.

The guards below pin the BEHAVIOUR, not the wiring:

    G1  resolution needs NO config key, and lands on the pre-registered artifact
    G2  the DEFAULT composition hands that path to the real builder
    G3  wiring it moves neither `Config.hash` nor `configs/base.yaml`
    G4  absent still degrades -- it does not crash
    G5  a specified-but-missing path is an ERROR, not a silent degrade
    G6  the checkpoint path never reaches the client seam

G3 and G6 are the two that would fail silently in production. G3 because a moved
hash detaches every shipped benchmark from its config; G6 because
`model_refs()` feeds `ExecutionTrace.selected_models`, which is published, and
`API_CONTRACT.md` section 7 records that v1 has no auth -- so wiring the encoder
is exactly what turns a previously-"unspecified" field into an absolute path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from core.config import load_config
from core.errors import ModelLoadError
from core.registry import RegistryState

from app import serving

#: The frozen hash the shipped artifacts were produced under.
FROZEN_CONFIG_HASH = "78f1e3700da15aa1"

#: The pre-registered identity of the recovered CROMA encoder.
CROMA_SHA256 = "0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63"
CROMA_BYTES = 777_563_846
CROMA_REVISION = "0dd28e3d633b"

#: The pre-registered identity of the recovered Phase 12 fusion head.
FUSION_HEAD_SHA256 = "785815729a3a39fc34dc41894efaf00d8739365d970a3f830a326e68ae888dab"
FUSION_HEAD_BYTES = 14_427_457

BASE_YAML_SHA256 = "88434f7f8f78e2b88d0d522e03ca86da25670a3f78b4f3c58c70b60044b80f89"

_REPO = Path(__file__).resolve().parents[2]
_BASE_YAML = _REPO / "configs" / "base.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _croma_cached_path() -> str | None:
    """The pinned snapshot, resolved OFFLINE ONLY.

    Deliberately not `resolve_checkpoint_path`: that falls through to the Hub on
    a cache miss, and a skip condition must not perform network I/O at
    collection time.
    """
    try:
        from huggingface_hub import hf_hub_download

        cfg = load_config()
        return hf_hub_download(
            cfg.get("croma.checkpoint_repo"),
            cfg.get("croma.checkpoint_file"),
            revision=cfg.get("croma.checkpoint_revision"),
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001 - "not cached here" is the whole point
        return None


requires_croma = pytest.mark.skipif(
    _croma_cached_path() is None,
    reason="pinned CROMA snapshot is not in the local Hub cache",
)


class _OpticalSarStub:
    """The minimal object the registry classifies for the `optical_sar` spec.

    Mirrors the real specialist's availability flags
    (`specialists/optical_sar/specialist.py:171` `has_encoder`, `:175`
    `has_head`) and the capability tuple the spec asserts against.
    """

    name = "optical_sar"
    version = "0.0.0-test"
    capabilities = ("optical_sar",)

    def __init__(self, *, has_encoder: bool = True, has_head: bool = True) -> None:
        self.has_encoder = has_encoder
        self.has_head = has_head


def _record(monkeypatch: pytest.MonkeyPatch, recorded: dict[str, Any], **flags: bool) -> None:
    """Replace the real builder with one that records its kwargs.

    The wiring imports the builder INSIDE the function body, so patching the
    module attribute is what the call site sees. Nothing here loads torch.
    """

    def _recording_builder(config: Any, **kwargs: Any) -> Any:
        recorded.update(kwargs)
        return _OpticalSarStub(**flags)

    monkeypatch.setattr(
        "specialists.optical_sar.specialist.build_optical_sar_specialist",
        _recording_builder,
    )


# ---------------------------------------------------------------------------
# G1a. Resolution needs no config key -- the mechanism
# ---------------------------------------------------------------------------
def test_resolution_needs_no_config_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pinned identity resolves WITHOUT `croma.checkpoint_path` existing.

    This is the tripwire for the tempting wrong fix. Adding the key to
    `configs/base.yaml` would also make serving work -- and would move
    `Config.hash`, detaching every shipped benchmark number from its config. If
    someone takes that route deliberately (an owner decision, not a patch), this
    assertion is the thing that makes them say so out loud.
    """
    from specialists.optical_sar.croma import (
        ENV_CROMA_CHECKPOINT,
        resolve_checkpoint_path,
    )

    monkeypatch.delenv(ENV_CROMA_CHECKPOINT, raising=False)

    cfg = load_config()
    assert cfg.get("croma.checkpoint_path") is None
    assert "checkpoint_path" not in _BASE_YAML.read_text(encoding="utf-8")

    _, source = resolve_checkpoint_path(cfg)
    assert source not in {"env", "config"}, (
        "resolution must come from the pinned identity, not from an explicit "
        f"path; got source={source!r}"
    )


# ---------------------------------------------------------------------------
# G1b. ...and it lands on the PRE-REGISTERED artifact
# ---------------------------------------------------------------------------
@requires_croma
def test_resolution_lands_on_the_pre_registered_croma_artifact() -> None:
    from specialists.optical_sar.croma import resolve_checkpoint_path

    path, source = resolve_checkpoint_path(load_config())

    assert source == "hub-cache"
    assert path is not None
    resolved = Path(path)
    assert resolved.exists()
    assert resolved.stat().st_size == CROMA_BYTES
    assert _sha256(resolved) == CROMA_SHA256
    assert CROMA_REVISION in resolved.as_posix()


# ---------------------------------------------------------------------------
# G2. The DEFAULT composition hands that path to the real builder
# ---------------------------------------------------------------------------
@requires_croma
def test_default_composition_wires_the_resolved_croma_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing guard: no override is applied by the caller here.

    `build_serving_registry` is called exactly as `build_serving_controller` and
    any deployment calls it -- with the config and nothing else.
    """
    from specialists.optical_sar.croma import resolve_checkpoint_path

    recorded: dict[str, Any] = {}
    _record(monkeypatch, recorded)

    registry = serving.build_serving_registry(load_config())
    entry = registry.build("optical_sar")

    expected_path, _ = resolve_checkpoint_path(load_config())
    assert recorded.get("checkpoint_path") == expected_path
    assert recorded.get("checkpoint_path") is not None
    assert Path(recorded["checkpoint_path"]).exists()
    assert entry.state is RegistryState.AVAILABLE
    assert entry.degraded is False


@requires_croma
def test_default_composition_wires_the_verified_fusion_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second half of "healthy".

    `has_head` (`specialist.py:175-187`) is False without a trained head, so the
    capability would stay DEGRADED even with the encoder loaded.
    """
    recorded: dict[str, Any] = {}
    _record(monkeypatch, recorded)

    serving.build_serving_registry(load_config()).build("optical_sar")

    assert recorded.get("head_path") == str(serving.FUSION_HEAD)
    assert Path(recorded["head_path"]).exists()
    assert Path(recorded["head_path"]).stat().st_size == FUSION_HEAD_BYTES
    assert _sha256(Path(recorded["head_path"])) == FUSION_HEAD_SHA256


# ---------------------------------------------------------------------------
# G3. Wiring it moves neither Config.hash nor configs/base.yaml
# ---------------------------------------------------------------------------
def test_wiring_optical_sar_does_not_move_the_config_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _record(monkeypatch, {})

    before = load_config().hash
    assert before == FROZEN_CONFIG_HASH

    registry = serving.build_serving_registry(load_config())
    registry.build("optical_sar")

    assert load_config().hash == before == FROZEN_CONFIG_HASH
    assert registry.config.hash == FROZEN_CONFIG_HASH
    assert _sha256(_BASE_YAML) == BASE_YAML_SHA256


# ---------------------------------------------------------------------------
# G4. Absent still degrades -- it does not crash
# ---------------------------------------------------------------------------
def test_absent_checkpoint_degrades_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}
    _record(monkeypatch, recorded, has_encoder=False, has_head=False)
    monkeypatch.setattr(
        "specialists.optical_sar.croma.resolve_checkpoint_path",
        lambda config=None: (None, "absent"),
    )

    registry = serving.build_serving_registry(load_config())
    entry = registry.build("optical_sar")  # must not raise

    assert recorded.get("checkpoint_path") is None
    assert entry.state is RegistryState.DEGRADED
    assert entry.degraded is True


# ---------------------------------------------------------------------------
# G5. A specified-but-missing path is an ERROR, not a silent degrade
# ---------------------------------------------------------------------------
def test_env_pointing_at_a_missing_checkpoint_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from specialists.optical_sar.croma import (
        ENV_CROMA_CHECKPOINT,
        resolve_checkpoint_path,
    )

    missing = tmp_path / "not_here" / "CROMA_base.pt"
    monkeypatch.setenv(ENV_CROMA_CHECKPOINT, str(missing))

    with pytest.raises(ModelLoadError) as excinfo:
        resolve_checkpoint_path(load_config())
    assert ENV_CROMA_CHECKPOINT in str(excinfo.value)


def test_config_pointing_at_a_missing_checkpoint_raises() -> None:
    from specialists.optical_sar.croma import resolve_checkpoint_path

    class _Cfg:
        """A config whose only explicit path is broken."""

        def get(self, key: str, default: Any = None) -> Any:
            return {
                "croma.checkpoint_path": r"C:\definitely\not\here\CROMA_base.pt",
                "croma.checkpoint_repo": "antofuller/CROMA",
                "croma.checkpoint_file": "CROMA_base.pt",
                "croma.checkpoint_revision": CROMA_REVISION,
            }.get(key, default)

    with pytest.raises(ModelLoadError):
        resolve_checkpoint_path(_Cfg())


# ---------------------------------------------------------------------------
# G6. The checkpoint path never reaches the client seam
# ---------------------------------------------------------------------------
def test_model_refs_publishes_only_the_basename() -> None:
    """`model_refs()` feeds `ExecutionTrace.selected_models`, which is published.

    `core/controller.py:689` folds these refs into the trace, and
    `API_CONTRACT.md` section 7 records that v1 has no auth. The encoder's own
    `describe()` keeps the full path for server-side diagnostics; the client
    sees the filename.
    """
    from specialists.optical_sar.specialist import OpticalSarSpecialist

    class _Encoder:
        def describe(self) -> dict[str, Any]:
            return {
                "checkpoint": (
                    r"C:\Users\anish\.cache\huggingface\hub"
                    r"\models--antofuller--CROMA\snapshots"
                    r"\0dd28e3d633bd6715856ae9890e8c49360040598\CROMA_base.pt"
                )
            }

    specialist = OpticalSarSpecialist(encoder=_Encoder(), head=None)
    refs = {ref["name"]: ref for ref in specialist.model_refs()}
    revision = refs["CROMA"]["revision"]

    assert revision == "CROMA_base.pt"
    assert "\\" not in revision
    assert ":" not in revision
    assert "anish" not in revision
    assert ".cache" not in revision


@requires_croma
def test_the_real_resolved_path_scrubs_to_a_path_free_revision() -> None:
    """The same guarantee, against the path this machine actually resolves."""
    from core.errors import scrub_paths
    from specialists.optical_sar.croma import resolve_checkpoint_path

    path, _ = resolve_checkpoint_path(load_config())
    scrubbed = scrub_paths(path)

    assert scrubbed == "CROMA_base.pt"
    assert "\\" not in scrubbed and "/" not in scrubbed


# ---------------------------------------------------------------------------
# G7. The public API still exports what the composition root promises
# ---------------------------------------------------------------------------
def test_optical_sar_wiring_is_reachable_from_the_public_package() -> None:
    import app

    assert app.FUSION_HEAD == serving.FUSION_HEAD
    assert "FUSION_HEAD" in serving.__all__


def test_the_fusion_head_is_the_pre_registered_artifact() -> None:
    assert serving.FUSION_HEAD.exists()
    assert serving.FUSION_HEAD.stat().st_size == FUSION_HEAD_BYTES
    assert _sha256(serving.FUSION_HEAD) == FUSION_HEAD_SHA256
