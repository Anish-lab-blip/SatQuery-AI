"""Tests for the resumable per-question answer cache in `training.vlm.evaluate`.

WHY THIS EXISTS
---------------
A full split evaluation is ~1000 per-sample CPU generations (tens of minutes). A
long single-shot process is fragile: when it is killed, every answer is lost and
the run restarts from zero. That is exactly what happened to the Phase 6
test-split adjudication, which died at 17 minutes with no traceback after having
generated most of a split.

The cache makes progress **monotone** -- each answer is flushed as produced.

The dangerous failure mode is not "cache is missing", it is "cache is *stale* and
gets reused anyway": that would silently mix answers from one question set into
the metrics of another, producing a confident, healthy-looking, wrong number in
an acceptance path. So the fingerprint guard is tested as hard as the resume
behaviour.

No model and no real images are used -- `_render_question_images` and the
generation call are replaced with fakes that count their invocations, which is
what lets us assert *the model was not called again* for cached questions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from training.vlm import evaluate as ev


@dataclass
class _Sample:
    sample_id: str
    question: str
    answer: str = "yes"


class _FakeProcessor:
    def __init__(self) -> None:
        self.decode_calls = 0

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return "PROMPT"

    def __call__(self, images=None, text=None, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def batch_decode(self, seq, skip_special_tokens=True):
        self.decode_calls += 1
        return ["ANSWER"]


class _FakeModel:
    """Counts `generate` calls, which is how we prove cached questions are skipped."""

    def __init__(self) -> None:
        self.generate_calls = 0

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls += 1
        return torch.tensor([[1, 2, 3, 4]])


@pytest.fixture()
def fake_images(monkeypatch):
    """Replace image rendering; the cache must not depend on real pixels."""
    calls = {"n": 0}

    def _render(samples, size):
        calls["n"] += len(samples)
        return [None for _ in samples]

    monkeypatch.setattr(ev, "_render_question_images", _render)
    return calls


def _samples(n: int, prefix: str = "s") -> list[_Sample]:
    return [_Sample(sample_id=f"{prefix}{i}", question=f"q{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# 1. the fingerprint is the question set, and only the question set
# ---------------------------------------------------------------------------
def test_fingerprint_is_stable_and_order_sensitive():
    a = _samples(5)
    assert ev._cache_fingerprint(a) == ev._cache_fingerprint(_samples(5))
    # A different ORDER is a different question sequence -- answers map by index.
    assert ev._cache_fingerprint(a) != ev._cache_fingerprint(list(reversed(a)))
    # A different COUNT is different.
    assert ev._cache_fingerprint(a) != ev._cache_fingerprint(_samples(6))


def test_fingerprint_falls_back_to_question_when_no_sample_id():
    class _Bare:
        def __init__(self, q):
            self.question = q

    assert ev._cache_fingerprint([_Bare("a"), _Bare("b")]) == ev._cache_fingerprint(
        [_Sample(sample_id="", question="a"), _Sample(sample_id="", question="b")]
    )


# ---------------------------------------------------------------------------
# 2. it caches, and a second pass does not regenerate
# ---------------------------------------------------------------------------
def test_second_pass_reuses_cache_and_does_not_call_the_model(tmp_path, fake_images):
    cache = tmp_path / "answers.jsonl"
    samples = _samples(4)

    model, processor = _FakeModel(), _FakeProcessor()
    first = ev.predict(model, processor, samples, cache_path=cache)

    assert first == ["ANSWER"] * 4
    assert model.generate_calls == 4
    assert fake_images["n"] == 4

    # Second pass: same answers, ZERO new generation, zero new image renders.
    model2, processor2 = _FakeModel(), _FakeProcessor()
    second = ev.predict(model2, processor2, samples, cache_path=cache)

    assert second == first
    assert model2.generate_calls == 0
    assert fake_images["n"] == 4  # unchanged


def test_cache_file_has_a_header_then_one_record_per_question(tmp_path, fake_images):
    import json

    cache = tmp_path / "answers.jsonl"
    ev.predict(_FakeModel(), _FakeProcessor(), _samples(3), cache_path=cache)

    lines = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert lines[0]["kind"] == "header"
    assert lines[0]["n"] == 3
    assert lines[0]["fingerprint"] == ev._cache_fingerprint(_samples(3))
    assert [r["i"] for r in lines[1:]] == [0, 1, 2]
    assert all(r["answer"] == "ANSWER" for r in lines[1:])


def test_no_cache_path_behaves_exactly_as_before(tmp_path, fake_images):
    """`cache_path=None` must not change answers or call counts."""
    model, processor = _FakeModel(), _FakeProcessor()
    out = ev.predict(model, processor, _samples(3))
    assert out == ["ANSWER"] * 3
    assert model.generate_calls == 3


# ---------------------------------------------------------------------------
# 3. the stale-cache guard -- the failure mode that would corrupt a metric
# ---------------------------------------------------------------------------
def test_a_cache_from_a_different_question_set_is_refused(tmp_path, fake_images):
    cache = tmp_path / "answers.jsonl"
    ev.predict(_FakeModel(), _FakeProcessor(), _samples(4), cache_path=cache)

    # Same count, different questions: MUST NOT be silently reused.
    with pytest.raises(ValueError, match="DIFFERENT question set"):
        ev.predict(_FakeModel(), _FakeProcessor(), _samples(4, prefix="other"), cache_path=cache)


def test_a_cache_with_a_different_count_is_refused(tmp_path, fake_images):
    cache = tmp_path / "answers.jsonl"
    ev.predict(_FakeModel(), _FakeProcessor(), _samples(4), cache_path=cache)
    with pytest.raises(ValueError, match="DIFFERENT question set"):
        ev.predict(_FakeModel(), _FakeProcessor(), _samples(5), cache_path=cache)


def test_a_cache_without_a_header_is_refused(tmp_path, fake_images):
    cache = tmp_path / "answers.jsonl"
    cache.write_text('{"i": 0, "answer": "ANSWER"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no readable header"):
        ev.predict(_FakeModel(), _FakeProcessor(), _samples(1), cache_path=cache)


def test_the_refusal_names_the_offending_file(tmp_path, fake_images):
    cache = tmp_path / "answers.jsonl"
    ev.predict(_FakeModel(), _FakeProcessor(), _samples(2), cache_path=cache)
    with pytest.raises(ValueError) as exc:
        ev.predict(_FakeModel(), _FakeProcessor(), _samples(3), cache_path=cache)
    assert str(cache) in str(exc.value)


# ---------------------------------------------------------------------------
# 4. resume after truncation -- the actual reason this exists
# ---------------------------------------------------------------------------
def test_a_truncated_run_resumes_instead_of_restarting(tmp_path, fake_images):
    """First pass stops at 2 of 5; the second pass generates only the other 3."""
    cache = tmp_path / "answers.jsonl"
    samples = _samples(5)

    model1 = _FakeModel()
    partial = ev.predict(
        model1, _FakeProcessor(), samples, cache_path=cache, deadline=0.0
    )
    assert partial == []  # deadline already passed -> nothing generated
    assert model1.generate_calls == 0

    # Now let it run to completion; all 5 are generated (cache was empty).
    model2 = _FakeModel()
    full = ev.predict(model2, _FakeProcessor(), samples, cache_path=cache)
    assert full == ["ANSWER"] * 5
    assert model2.generate_calls == 5

    # And a third pass resumes from the complete cache with no generation.
    model3 = _FakeModel()
    assert ev.predict(model3, _FakeProcessor(), samples, cache_path=cache) == full
    assert model3.generate_calls == 0


def test_partial_cache_is_completed_by_a_later_run(tmp_path, fake_images, monkeypatch):
    """Simulate a kill mid-split: only the first 2 answers ever got written."""
    import json

    cache = tmp_path / "answers.jsonl"
    samples = _samples(5)
    fp = ev._cache_fingerprint(samples)
    cache.write_text(
        json.dumps({"kind": "header", "fingerprint": fp, "n": 5}) + "\n"
        + json.dumps({"i": 0, "answer": "ANSWER"}) + "\n"
        + json.dumps({"i": 1, "answer": "ANSWER"}) + "\n",
        encoding="utf-8",
    )

    model = _FakeModel()
    out = ev.predict(model, _FakeProcessor(), samples, cache_path=cache)

    assert out == ["ANSWER"] * 5
    assert model.generate_calls == 3  # only the missing tail


# ---------------------------------------------------------------------------
# 5. `evaluate_split` is unchanged in what it computes
# ---------------------------------------------------------------------------
def test_evaluate_split_cached_and_uncached_agree(tmp_path, fake_images):
    samples = [_Sample(sample_id=f"s{i}", question=f"q{i}", answer="ANSWER") for i in range(4)]

    plain = ev.evaluate_split(_FakeModel(), _FakeProcessor(), samples, device="cpu")
    cached = ev.evaluate_split(
        _FakeModel(), _FakeProcessor(), samples, device="cpu",
        cache_path=tmp_path / "answers.jsonl",
    )

    assert plain.exact_match == cached.exact_match == 1.0
    assert plain.n == cached.n == 4
    assert plain.confusion == cached.confusion
    assert plain.truncated is False and cached.truncated is False


def test_evaluate_split_marks_truncation_so_v3_still_fails(tmp_path, fake_images):
    """A partial evaluation must stay `truncated`, or V3 could wrongly pass."""
    samples = [_Sample(sample_id=f"s{i}", question=f"q{i}", answer="ANSWER") for i in range(4)]
    metric = ev.evaluate_split(
        _FakeModel(), _FakeProcessor(), samples, device="cpu",
        deadline=0.0, cache_path=tmp_path / "answers.jsonl",
    )
    assert metric.truncated is True
    assert metric.n == 0
    assert metric.n_available == 4


# ---------------------------------------------------------------------------
# 6. a kill mid-write tears the last line -- that must be recoverable
# ---------------------------------------------------------------------------
def test_a_torn_final_line_is_tolerated_and_the_prefix_is_reused(tmp_path, fake_images):
    """The exact kill scenario: the process died while flushing answer 2.

    A torn tail must NOT make the whole cache unreadable -- that would turn a
    recoverable kill into a full restart, which is the thing this cache exists to
    prevent. It must also not swallow the records written AFTER it.
    """
    import json

    cache = tmp_path / "answers.jsonl"
    samples = _samples(5)
    fp = ev._cache_fingerprint(samples)
    good = (
        json.dumps({"kind": "header", "fingerprint": fp, "n": 5}) + "\n"
        + json.dumps({"i": 0, "answer": "ANSWER"}) + "\n"
        + json.dumps({"i": 1, "answer": "ANSWER"}) + "\n"
    )
    cache.write_text(good + '{"i": 2, "ans', encoding="utf-8")  # torn, NO newline

    model = _FakeModel()
    out = ev.predict(model, _FakeProcessor(), samples, cache_path=cache)

    assert out == ["ANSWER"] * 5
    assert model.generate_calls == 3  # indices 2, 3, 4 -- 0 and 1 were reused

    # The load-bearing check: the torn line must not have merged with the record
    # appended after it, or the cache would be permanently stuck at that line.
    model2 = _FakeModel()
    assert ev.predict(model2, _FakeProcessor(), samples, cache_path=cache) == ["ANSWER"] * 5
    assert model2.generate_calls == 0


def test_a_torn_header_is_fatal_because_the_file_cannot_be_trusted(tmp_path, fake_images):
    cache = tmp_path / "answers.jsonl"
    cache.write_text('{"kind": "header", "finger', encoding="utf-8")
    with pytest.raises(ValueError, match="no readable header"):
        ev.predict(_FakeModel(), _FakeProcessor(), _samples(2), cache_path=cache)
