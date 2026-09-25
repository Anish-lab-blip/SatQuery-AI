"""Wiring the trained RemoteCLIP grounding head into production (work order §5).

THE GAP THESE TESTS CLOSE
-------------------------
Everything needed to serve the trained head already existed and was already
correct:

  * `core/registry.py` declared `grounding_head.head_path` as an optional config
    key for the `grounding` specialist;
  * `build_grounding_specialist` accepted a `head_path` argument;
  * `GroundingSpecialist` implemented the head decode, the zero-shot decode, and
    the `degraded` flag that distinguishes them;
  * `artifacts/grounding/remoteclip_grounding_v001/head.pt` was on disk.

Measured 2026-09-23: `head_path` appears nowhere in `configs/base.yaml` or
`core/config.py`, and no call site passes it. So the registry supplied no kwarg,
the builder defaulted to `None`, and **production silently ran the zero-shot
baseline while a trained head sat unused**. Every individual piece was right; the
wire between them was missing.

WHAT WAS CHANGED
----------------
`head_path=None` now means "use the shipped head" (`DEFAULT_HEAD_PATH`) rather
than "no head". The default is a code constant, NOT a `base.yaml` entry, because
`base.yaml` is hashed into the frozen config identity -- see
`test_the_frozen_config_hash_is_untouched` below, which pins that decision.

The four outcomes are kept distinct, because they need four different fixes:

    no path configured + shipped head absent  -> source "shipped_default"
    configured path does not exist            -> source "configured"
    exists but will not load                  -> source "configured"/"shipped_default"
    exists but the wrong architecture         -> refused, with both widths named

and the reason travels into the result, so a trained head is never claimed when
one is not loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import load_config
from specialists.grounding.specialist import (
    DEFAULT_HEAD_PATH,
    GroundingSpecialist,
    HeadLoadReport,
    load_grounding_head,
)

#: The frozen config identity. `Config.hash` is sha256 of the YAML registry,
#: truncated to 16 characters.
FROZEN_CONFIG_HASH = "78f1e3700da15aa1"

head_present = pytest.mark.skipif(
    not DEFAULT_HEAD_PATH.exists(),
    reason=f"shipped grounding head not present at {DEFAULT_HEAD_PATH}",
)


class StubEncoder:
    """Minimal RemoteCLIP stand-in. Only the call surface matters here."""

    model_name = "stub-remoteclip"
    checkpoint_path = "stub-checkpoint.pt"


class StubHead:
    def eval(self) -> None:
        return None

    def num_parameters(self) -> int:
        return 1_052_677


# ---------------------------------------------------------------------------
# The frozen config identity
# ---------------------------------------------------------------------------
class TestFrozenInvariants:
    def test_the_frozen_config_hash_is_untouched(self):
        """Pins the reason `DEFAULT_HEAD_PATH` is a code constant.

        `base.yaml` is hashed into `Config.hash`, which run manifests and
        published numbers cite. Wiring the head by editing `base.yaml` would have
        been the obvious fix and would have moved this value. If this test fails,
        something edited the config registry -- which may be intended, but is
        never incidental.
        """
        assert load_config().hash == FROZEN_CONFIG_HASH

    def test_the_head_path_config_key_is_declared_as_an_override(self):
        """The registry key exists and keeps its original meaning.

        `default_specs()` is the registry's own accessor -- there is no module
        level `SPECIALIST_SPECS` constant to import.
        """
        from core.registry import default_specs

        spec = next(s for s in default_specs() if s.name == "grounding")
        assert spec.optional_config_keys.get("head_path") == "grounding_head.head_path"

    def test_the_shipped_head_path_is_not_hard_coded_in_the_config(self):
        """The default lives in code, so the config stays an override surface."""
        config = load_config()
        assert config.get("grounding_head.head_path", None) is None


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------
class TestHeadLoader:
    @head_present
    def test_none_resolves_to_the_shipped_head(self):
        head, report = load_grounding_head(None, device="cpu")
        assert head is not None
        assert report.loaded is True
        assert report.source == "shipped_default"
        assert report.resolved_path == str(DEFAULT_HEAD_PATH)
        assert report.reason is None

    @head_present
    def test_the_loaded_head_has_the_architecture_the_config_declares(self):
        """`build_head` guards this at training time; this guards it at serve time."""
        config = load_config()
        head, report = load_grounding_head(
            None,
            device="cpu",
            feature_dim=int(config.get("grounding_head.feature_dim", 2048)),
        )
        assert report.loaded is True
        assert report.detail["feature_dim"] == int(
            config.get("grounding_head.feature_dim")
        )
        assert report.detail["hidden_dim"] == int(config.get("grounding_head.hidden_dim"))
        assert report.detail["n_parameters"] > 0

    @head_present
    def test_a_feature_dim_mismatch_is_refused_with_both_widths_named(self):
        head, report = load_grounding_head(None, device="cpu", feature_dim=1024)
        assert head is None
        assert report.loaded is False
        assert "2048" in report.reason and "1024" in report.reason
        assert "similarity step" in report.reason

    def test_a_configured_path_that_does_not_exist_is_reported_as_configured(self, tmp_path):
        """`configured` and `shipped_default` need different actions."""
        missing = tmp_path / "nope" / "head.pt"
        head, report = load_grounding_head(missing, device="cpu")
        assert head is None
        assert report.loaded is False
        assert report.source == "configured"
        assert "does not exist" in report.reason
        assert report.detail["exists"] is False

    def test_a_corrupt_checkpoint_is_refused_and_names_the_error(self, tmp_path):
        bad = tmp_path / "head.pt"
        bad.write_bytes(b"this is not a torch checkpoint")
        head, report = load_grounding_head(bad, device="cpu")
        assert head is None
        assert report.loaded is False
        assert "could not be loaded" in report.reason
        assert report.detail["exists"] is True
        assert report.detail["error_type"]

    def test_a_checkpoint_that_is_not_a_dict_is_refused(self, tmp_path):
        """A valid pickle with the wrong shape must degrade, not crash."""
        import torch

        bad = tmp_path / "head.pt"
        torch.save([1, 2, 3], bad)
        head, report = load_grounding_head(bad, device="cpu")
        assert head is None
        assert report.loaded is False

    def test_the_loader_never_raises_for_a_bad_checkpoint(self, tmp_path):
        """The caller's fallback is zero-shot; a crash here would remove it."""
        bad = tmp_path / "head.pt"
        bad.write_bytes(b"\x00\x01\x02")
        try:
            load_grounding_head(bad, device="cpu")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"load_grounding_head raised {type(exc).__name__}: {exc}")

    def test_the_report_serialises(self, tmp_path):
        _head, report = load_grounding_head(tmp_path / "absent.pt", device="cpu")
        payload = report.to_dict()
        for key in (
            "requested_path",
            "resolved_path",
            "loaded",
            "source",
            "reason",
            "detail",
            "note",
        ):
            assert key in payload
        assert "zero-shot" in payload["note"]


# ---------------------------------------------------------------------------
# The specialist reports which decode it used
# ---------------------------------------------------------------------------
class TestSpecialistReporting:
    def test_a_specialist_with_a_head_is_not_degraded(self):
        specialist = GroundingSpecialist(encoder=StubEncoder(), head=StubHead())
        assert specialist.has_head is True
        breakdown = specialist._confidence_for({}, used_head=True)
        assert breakdown.degraded is False
        assert breakdown.degradation_reason is None

    def test_a_specialist_without_a_head_is_degraded_and_says_why(self):
        report = HeadLoadReport(
            requested_path=None,
            resolved_path=str(DEFAULT_HEAD_PATH),
            loaded=False,
            source="shipped_default",
            reason="the shipped head is absent",
        )
        specialist = GroundingSpecialist(
            encoder=StubEncoder(), head=None, head_report=report
        )
        assert specialist.has_head is False
        breakdown = specialist._confidence_for({}, used_head=False)
        assert breakdown.degraded is True
        assert "zero-shot fallback" in breakdown.degradation_reason
        assert "shipped head is absent" in breakdown.degradation_reason

    def test_the_fallback_reason_distinguishes_the_causes(self):
        """One generic string for four causes is the thing this replaces."""
        reasons = {
            "missing": "the configured head_path /x/head.pt does not exist",
            "invalid": "the head at /x/head.pt exists but could not be loaded",
            "absent_default": "no head_path was configured and the shipped head is absent",
        }
        seen = set()
        for name, reason in reasons.items():
            specialist = GroundingSpecialist(
                encoder=StubEncoder(),
                head=None,
                head_report=HeadLoadReport(
                    requested_path=None,
                    resolved_path="/x/head.pt",
                    loaded=False,
                    source="configured",
                    reason=reason,
                ),
            )
            text = specialist._confidence_for({}, used_head=False).degradation_reason
            assert reason in text, name
            seen.add(text)
        assert len(seen) == len(reasons), "the reasons must not collapse to one string"

    def test_a_default_report_is_synthesised_when_none_is_supplied(self):
        """Constructing without a report must still yield a usable reason."""
        specialist = GroundingSpecialist(encoder=StubEncoder(), head=None)
        assert specialist.head_report.loaded is False
        assert specialist.head_report.reason
        assert specialist._confidence_for({}, used_head=False).degradation_reason

    def test_an_injected_head_is_reported_as_injected(self):
        """`injected` means a caller supplied the object, so no path was read."""
        specialist = GroundingSpecialist(encoder=StubEncoder(), head=StubHead())
        assert specialist.head_report.source == "injected"
        assert specialist.head_report.loaded is True

    @head_present
    def test_the_default_report_reflects_a_real_load(self):
        """End-to-end: the shipped head must make `has_head` True."""
        head, report = load_grounding_head(None, device="cpu")
        specialist = GroundingSpecialist(
            encoder=StubEncoder(), head=head, head_report=report
        )
        assert specialist.has_head is True
        assert specialist.head_report.source == "shipped_default"
        assert specialist._confidence_for({}, used_head=True).degraded is False
