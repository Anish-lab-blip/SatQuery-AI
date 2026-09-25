"""QA scratch probe (NOT collected by pytest -- underscore prefix).

Measures the supervised-token count of a valid answer so the refusal tests can
pin "same count as before". Writes its findings to tests/unit/_qa_probe_out.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import transformers  # noqa: E402

from training.vlm.formatting import format_example  # noqa: E402

QUESTION = "Is water present in this image?"


def main() -> None:
    processor = transformers.AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM-500M-Instruct", size={"longest_edge": 512}
    )
    lines = []
    for answer in ("Yes.", "No.", "Yes", "yes"):
        ex = format_example(
            processor, question=QUESTION, answer=answer, max_seq_length=512
        )
        lines.append(
            f"answer={answer!r} prompt_length={ex.prompt_length} "
            f"full_length={ex.full_length} n_supervised={ex.n_supervised_tokens}"
        )
    out = Path(__file__).resolve().parent / "_qa_probe_out.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
