"""Shared fixtures for the Phase 6 VLM unit tests.

The only expensive resource these tests need is the SmolVLM **processor** -- not
the model. The masking and collation contracts can only be tested against the
real chat template and the real image expander; a hand-built stub would test the
stub. It is loaded once per session.

It is pinned exactly the way the trainer pins it
(`specialists/vqa/model.py::SmolVLM._build_processor` passes
`size={"longest_edge": 512}`), and the pin is load-bearing: an unpinned
processor upscales a 512 px tile 4x and then splits it into 4x4 sub-images + 1
overview = 17 images and ~1147 prompt tokens (finding F5-2, measured), which is
a different token distribution from the one the trainer sees and would make the
collation tests assert against the wrong shape.
"""

from __future__ import annotations

from typing import Any

import pytest

#: The checkpoint the frozen registry names (`configs/base.yaml`: vlm.checkpoint).
SMOLVLM_CHECKPOINT = "HuggingFaceTB/SmolVLM-500M-Instruct"

#: The F5-2 resolution pin, mirrored from the serving loader.
PROCESSOR_LONGEST_EDGE = 512


@pytest.fixture(scope="session")
def smolvlm_processor() -> Any:
    """The pinned SmolVLM processor. Loaded once; no model is loaded."""
    transformers = pytest.importorskip("transformers")
    return transformers.AutoProcessor.from_pretrained(
        SMOLVLM_CHECKPOINT, size={"longest_edge": PROCESSOR_LONGEST_EDGE}
    )
