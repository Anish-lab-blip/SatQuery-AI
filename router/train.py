"""SatQuery AI — intent router training.

Training strategy, and why it is this cheap (finding F4-2):

    The encoder is FROZEN. Embeddings are therefore a pure function of the
    query text. So we embed the whole corpus ONCE (measured: 0.118 s / 64
    queries on CPU), cache the vectors to disk, and train the 50,822-parameter
    adapter on the cached matrix (measured: 20 epochs / 4,096 vectors in 0.28 s).

    Router training needs NO GPU. The plan's Kaggle budget ("Router | CPU/T4 |
    <1 h") is roughly three orders of magnitude pessimistic. Phase 4 runs and
    completes locally.

Split discipline (finding F4-3):

    Groups are template ids and curated families. Splitting by example would
    put "show me the water body" in train and "show me the road" in val — same
    template, one token apart — and report a fake accuracy. `split_by_group`
    mirrors `evaluation.leakage.assign_splits_by_scene` deliberately, and the
    train function REFUSES to proceed if the split report is not clean.

Loss:

    task       cross-entropy, class-weighted (classes are imbalanced)
    modality   cross-entropy
    binary x3  BCE-with-logits, one logit per head

    Weights live in config under `router.training`.

Usage:
    python scripts/train_router.py --smoke      # 3 epochs, tiny corpus
    python scripts/train_router.py              # full run
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from core.errors import RoutingError
from router.adapter import IntentAdapter
from router.classifier import save_adapter
from router.dataset import (
    HARD_NEGATIVE_PREFIX,
    RouterCorpus,
    RouterExample,
    build_corpus,
    split_by_group,
    split_leakage_report,
)
from router.label_space import (
    BINARY_HEADS,
    MODALITY_CLASSES,
    MODALITY_TO_INDEX,
    NUM_MODALITIES,
    NUM_TASKS,
    TASK_CLASSES,
    TASK_TO_INDEX,
)

CORPUS_CACHE_VERSION = "v1"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SplitMetrics:
    """Per-split evaluation results."""

    name: str
    n: int
    task_accuracy: float
    modality_accuracy: float
    binary_accuracy: dict[str, float]
    combined_accuracy: float
    macro_f1_task: float
    per_task_recall: dict[str, float] = field(default_factory=dict)
    #: How many test examples each class actually had. A recall without
    #: support is a number that means nothing; they are reported together.
    task_support: dict[str, int] = field(default_factory=dict)
    hard_negative_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "task_accuracy": round(self.task_accuracy, 4),
            "modality_accuracy": round(self.modality_accuracy, 4),
            "binary_accuracy": {k: round(v, 4) for k, v in self.binary_accuracy.items()},
            "combined_accuracy": round(self.combined_accuracy, 4),
            "macro_f1_task": round(self.macro_f1_task, 4),
            "per_task_recall": {k: round(v, 4) for k, v in self.per_task_recall.items()},
            "task_support": dict(self.task_support),
            "hard_negative_accuracy": (
                None if self.hard_negative_accuracy is None
                else round(self.hard_negative_accuracy, 4)
            ),
        }


@dataclass
class TrainingResult:
    """Everything the caller needs to decide the next step."""

    artifact_dir: Path
    metadata: dict[str, Any]
    train_metrics: SplitMetrics
    val_metrics: SplitMetrics
    test_metrics: SplitMetrics
    split_report: dict[str, Any]
    corpus_hash: str
    duration_seconds: float
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def unmeasured_test_tasks(self) -> list[str]:
        """Task classes with no test examples. The gate cannot speak for these."""
        return [
            task for task in TASK_CLASSES
            if self.test_metrics.task_support.get(task, 0) == 0
        ]

    @property
    def gate2_passed(self) -> bool:
        """Gate 2 passes only if EVERY task class was measured AND cleared 0.95.

        Without the first condition a class can be absent from the test set and
        the gate reports a pass over a model that was never asked about it.
        """
        return not self.unmeasured_test_tasks and self.test_metrics.task_accuracy >= 0.95

    def summary(self) -> str:
        lines = [
            "=" * 66,
            "SATQUERY AI - ROUTER TRAINING COMPLETE",
            "=" * 66,
            f"corpus          : {self.metadata['corpus']['total']} examples",
            f"groups          : {self.metadata['corpus']['groups']}",
            f"corpus hash     : {self.corpus_hash[:16]}",
            f"encoder         : {self.metadata['encoder']['model']}",
            f"                  @ {self.metadata['encoder']['revision']} "
            f"(frozen, {self.metadata['encoder']['parameters']:,} params)",
            f"adapter params  : {self.metadata['adapter']['num_parameters']:,}",
            f"epochs run      : {len(self.history)}",
            f"duration        : {self.duration_seconds:.1f} s",
            "",
            "SPLIT SIZES",
            f"  train {self.split_report['sizes']['train']:>5}  "
            f"val {self.split_report['sizes']['val']:>5}  "
            f"test {self.split_report['sizes']['test']:>5}",
            f"  leakage clean : {self.split_report['clean']}",
            "",
            "METRICS",
            f"  {'split':<7} {'task':>7} {'modal':>7} {'spat':>7} {'temp':>7} "
            f"{'lang':>7} {'comb':>7} {'macroF1':>8}",
        ]
        for m in (self.train_metrics, self.val_metrics, self.test_metrics):
            lines.append(
                f"  {m.name:<7} {m.task_accuracy:>7.3f} {m.modality_accuracy:>7.3f} "
                f"{m.binary_accuracy.get('spatial_output', 0):>7.3f} "
                f"{m.binary_accuracy.get('temporal', 0):>7.3f} "
                f"{m.binary_accuracy.get('language_output', 0):>7.3f} "
                f"{m.combined_accuracy:>7.3f} {m.macro_f1_task:>8.3f}"
            )
        if self.test_metrics.hard_negative_accuracy is not None:
            lines.append("")
            lines.append(
                f"HARD-NEGATIVE ACCURACY (test) : "
                f"{self.test_metrics.hard_negative_accuracy:.3f}"
            )
        lines.append("")
        lines.append("PER-TASK RECALL (test)            recall       n")
        for task in TASK_CLASSES:
            n_task = self.test_metrics.task_support.get(task, 0)
            if n_task == 0:
                lines.append(f"  {task:<12} {'n/a':>18} {n_task:>6}  (no test examples)")
            else:
                recall = self.test_metrics.per_task_recall.get(task, 0.0)
                lines.append(f"  {task:<12} {recall:>18.3f} {n_task:>6}")
        lines.append("")
        lines.append("GATE 2 ROUTER ACCEPTANCE: task accuracy >= 0.95, all classes measured")
        measured = len(TASK_CLASSES) - len(self.unmeasured_test_tasks)
        lines.append(f"  classes with test support : {measured}/{len(TASK_CLASSES)}")
        if self.unmeasured_test_tasks:
            lines.append(
                f"  classes NOT measurable    : {', '.join(self.unmeasured_test_tasks)}"
            )
        acc = self.test_metrics.task_accuracy
        if self.unmeasured_test_tasks:
            lines.append(
                f"  -> NOT MEASURABLE ({acc:.3f} over the {measured} classes that "
                f"have test data)"
            )
        else:
            verdict = "PASS" if acc >= 0.95 else "BELOW TARGET"
            lines.append(f"  -> {verdict} ({acc:.3f})")
        lines.append("")
        lines.append(f"artifact        : {self.artifact_dir}")
        lines.append("=" * 66)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def _corpus_fingerprint(corpus: RouterCorpus, encoder_id: str, max_length: int) -> str:
    payload = {
        "version": CORPUS_CACHE_VERSION,
        "encoder": encoder_id,
        "max_length": max_length,
        "texts": sorted(e.text for e in corpus),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def embed_corpus_cached(
    corpus: RouterCorpus,
    encoder,
    cache_dir: str | Path,
    batch_size: int = 64,
    force: bool = False,
) -> np.ndarray:
    """Embed the corpus, reusing a disk cache when the fingerprint matches.

    The cache key includes the encoder id AND the corpus text set, so changing
    either invalidates it. A stale cache silently training on the wrong vectors
    would be worse than no cache.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    encoder_id = f"{encoder.model_name}@{encoder.revision}"
    fingerprint = _corpus_fingerprint(corpus, encoder_id, encoder.max_length)
    vectors_file = cache_path / f"embeddings_{fingerprint[:16]}.npy"
    meta_file = cache_path / f"embeddings_{fingerprint[:16]}.json"

    if vectors_file.exists() and meta_file.exists() and not force:
        try:
            cached = np.load(vectors_file)
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if cached.shape[0] == len(corpus) and meta.get("fingerprint") == fingerprint:
                return cached.astype(np.float32)
        except Exception:  # noqa: BLE001 - a bad cache is a miss, not an error
            pass

    started = time.time()
    vectors = encoder.encode(corpus.texts(), batch_size=batch_size, normalize=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    np.save(vectors_file, vectors)
    meta_file.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "encoder": encoder_id,
                "max_length": encoder.max_length,
                "count": int(vectors.shape[0]),
                "dim": int(vectors.shape[1]),
                "seconds": round(time.time() - started, 3),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return vectors


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


def _to_tensors(
    examples: list[RouterExample],
    embeddings: np.ndarray,
    device: str,
) -> dict[str, torch.Tensor]:
    task_idx = torch.tensor(
        [TASK_TO_INDEX[e.task] for e in examples], dtype=torch.long, device=device
    )
    modality_idx = torch.tensor(
        [MODALITY_TO_INDEX[e.modality] for e in examples], dtype=torch.long, device=device
    )
    binaries = torch.tensor(
        [[1.0 if getattr(e, h) else 0.0 for h in BINARY_HEADS] for e in examples],
        dtype=torch.float32,
        device=device,
    )
    x = torch.from_numpy(embeddings).to(device)
    return {"x": x, "task": task_idx, "modality": modality_idx, "binary": binaries}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _macro_f1(y_true: list[int], y_pred: list[int], n_classes: int) -> float:
    """Macro-averaged F1. Implemented locally to avoid a sklearn dependency in
    the training path (sklearn is installed for dev, not required at runtime)."""
    if not y_true:
        return 0.0
    f1s: list[float] = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def _recall_per_class(
    y_true: list[int], y_pred: list[int], n_classes: int
) -> tuple[list[float], list[int]]:
    """Per-class recall AND per-class support.

    Support is returned alongside recall because a class with zero test
    examples has a recall of 0.0 that means nothing. Returning them together
    makes that impossible to misread: `vqa 0.000` without support is exactly
    the kind of number that gets quoted as a failure when it is an absence.
    """
    recalls: list[float] = []
    support: list[int] = []
    for c in range(n_classes):
        total = sum(1 for t in y_true if t == c)
        hit = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        support.append(total)
        recalls.append(hit / total if total else 0.0)
    return recalls, support


@torch.no_grad()
def evaluate_split(
    model: IntentAdapter,
    examples: list[RouterExample],
    embeddings: np.ndarray,
    name: str,
    device: str,
    batch_size: int = 128,
) -> SplitMetrics:
    """Evaluate the adapter over one split."""
    model.eval()
    if not examples:
        return SplitMetrics(
            name=name, n=0, task_accuracy=0.0, modality_accuracy=0.0,
            binary_accuracy={h: 0.0 for h in BINARY_HEADS},
            combined_accuracy=0.0, macro_f1_task=0.0,
            task_support={task: 0 for task in TASK_CLASSES},
        )

    tensors = _to_tensors(examples, embeddings, device)
    n = len(examples)

    task_pred: list[int] = []
    modality_pred: list[int] = []
    binary_pred: dict[str, list[int]] = {h: [] for h in BINARY_HEADS}

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        out = model(tensors["x"][start:stop])
        task_pred.extend(torch.argmax(out.task_logits, dim=-1).cpu().tolist())
        modality_pred.extend(torch.argmax(out.modality_logits, dim=-1).cpu().tolist())
        for head in BINARY_HEADS:
            logits = out.binary_logit(head)
            binary_pred[head].extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())

    task_true = tensors["task"].cpu().tolist()
    modality_true = tensors["modality"].cpu().tolist()

    task_acc = sum(1 for t, p in zip(task_true, task_pred) if t == p) / n
    modality_acc = sum(1 for t, p in zip(modality_true, modality_pred) if t == p) / n

    binary_acc: dict[str, float] = {}
    for i, head in enumerate(BINARY_HEADS):
        true_vals = tensors["binary"][:, i].cpu().tolist()
        pred_vals = binary_pred[head]
        binary_acc[head] = (
            sum(1 for t, p in zip(true_vals, pred_vals) if int(t) == p) / n
        )

    # Combined: every head must be right for an example to count.
    combined_hits = 0
    for i in range(n):
        ok = task_pred[i] == task_true[i] and modality_pred[i] == modality_true[i]
        if ok:
            for j, head in enumerate(BINARY_HEADS):
                if int(tensors["binary"][i, j].item()) != binary_pred[head][i]:
                    ok = False
                    break
        combined_hits += int(ok)

    recalls, support = _recall_per_class(task_true, task_pred, NUM_TASKS)

    # Hard negatives: examples whose group starts with "hn_" (curated) — these
    # are the one-token-difference pairs the router exists to get right.
    hn_examples = [e for e in examples if e.group.startswith(HARD_NEGATIVE_PREFIX)]
    hn_acc: float | None = None
    if hn_examples:
        hn_idx = [i for i, e in enumerate(examples) if e.group.startswith(HARD_NEGATIVE_PREFIX)]
        hits = sum(1 for i in hn_idx if task_pred[i] == task_true[i])
        hn_acc = hits / len(hn_idx)

    return SplitMetrics(
        name=name,
        n=n,
        task_accuracy=task_acc,
        modality_accuracy=modality_acc,
        binary_accuracy=binary_acc,
        combined_accuracy=combined_hits / n,
        macro_f1_task=_macro_f1(task_true, task_pred, NUM_TASKS),
        per_task_recall={
            TASK_CLASSES[i]: recalls[i] for i in range(NUM_TASKS) if support[i] > 0
        },
        task_support={TASK_CLASSES[i]: support[i] for i in range(NUM_TASKS)},
        hard_negative_accuracy=hn_acc,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_router(
    corpus: RouterCorpus | None = None,
    encoder=None,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    task_loss_weight: float = 1.0,
    modality_loss_weight: float = 0.3,
    binary_loss_weight: float = 0.5,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
    artifact_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    config_hash: str | None = None,
    n_template_repeats: int = 1,
    val_ratio: float = 0.15,
    hard_negatives_to_test: bool = True,
    verbose: bool = True,
) -> TrainingResult:
    """Train the intent adapter on frozen embeddings.

    `encoder` may be None, in which case a deterministic hash-based
    pseudo-embedding is used. That path exists ONLY so the training loop can be
    tested without downloading a model; it is not a real router and the
    artifact it produces is marked `encoder_type: "stub"`.
    """
    started = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -- corpus ------------------------------------------------------------
    if corpus is None:
        corpus = build_corpus(n_template_repeats=n_template_repeats, seed=seed)

    problems = corpus.validate()
    if problems:
        raise RoutingError(
            "router corpus failed validation:\n  - " + "\n  - ".join(problems)
        )

    corpus, conflicts = corpus.dedupe()
    if conflicts:
        raise RoutingError(
            "router corpus contains labelling conflicts (same text, two tasks):\n  - "
            + "\n  - ".join(conflicts)
        )

    # -- split, with the leakage guard PROVEN to have fired -----------------
    split = split_by_group(
        corpus,
        seed=seed,
        val_ratio=val_ratio,
        hard_negatives_to_test=hard_negatives_to_test,
    )
    report = split_leakage_report(split)
    if not report["clean"]:
        raise RoutingError(
            "router split failed the group-leakage audit; refusing to train: "
            f"{report['groups_across_splits']}"
        )

    train_ex, val_ex, test_ex = split["train"], split["val"], split["test"]
    if not train_ex or not val_ex or not test_ex:
        raise RoutingError(
            f"split produced an empty partition: train={len(train_ex)} "
            f"val={len(val_ex)} test={len(test_ex)}"
        )

    # -- embeddings --------------------------------------------------------
    if encoder is not None:
        embeddings = embed_corpus_cached(
            corpus, encoder, cache_dir or Path("artifacts/router/cache"),
            force=False,
        )
        input_dim = encoder.embedding_dim
        encoder_meta = {
            "model": encoder.model_name,
            "revision": encoder.revision,
            "max_length": encoder.max_length,
            "parameters": encoder.num_parameters,
            "type": "frozen_sentence_transformer",
        }
        encoder_type = "frozen_sentence_transformer"
    else:
        embeddings = _stub_embeddings(corpus, dim=384)
        input_dim = 384
        encoder_meta = {
            "model": "stub",
            "revision": "none",
            "max_length": 0,
            "parameters": 0,
            "type": "stub",
        }
        encoder_type = "stub"

    corpus_hash = _corpus_fingerprint(
        corpus, encoder_meta["model"] + "@" + str(encoder_meta["revision"]), 0
    )

    # -- index bookkeeping: embeddings follow corpus order -----------------
    text_to_row = {e.text: i for i, e in enumerate(corpus)}
    all_ex = corpus.examples

    def rows_for(examples: list[RouterExample]) -> np.ndarray:
        idx = [text_to_row[e.text] for e in examples]
        return embeddings[np.asarray(idx, dtype=np.int64)]

    train_x = rows_for(train_ex)
    val_x = rows_for(val_ex)
    test_x = rows_for(test_ex)

    # -- model -------------------------------------------------------------
    model = IntentAdapter(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_tasks=NUM_TASKS,
        num_modalities=NUM_MODALITIES,
    ).to(device)

    # Class weights: inverse frequency on the TRAIN split only. Computing them
    # on the full corpus would leak val/test label distribution into training.
    train_task_counts = np.zeros(NUM_TASKS, dtype=np.float64)
    for e in train_ex:
        train_task_counts[TASK_TO_INDEX[e.task]] += 1
    counts = np.maximum(train_task_counts, 1.0)
    class_weights = torch.tensor(
        (counts.sum() / (NUM_TASKS * counts)), dtype=torch.float32, device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    ce = nn.CrossEntropyLoss(weight=class_weights)
    ce_mod = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    train_tensors = _to_tensors(train_ex, train_x, device)
    n_train = len(train_ex)
    history: list[dict[str, float]] = []
    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    rng = random.Random(seed)

    for epoch in range(epochs):
        model.train()
        order = list(range(n_train))
        rng.shuffle(order)

        epoch_loss = 0.0
        batches = 0

        for start in range(0, n_train, batch_size):
            idx = torch.tensor(order[start:start + batch_size], dtype=torch.long, device=device)
            out = model(train_tensors["x"][idx])

            loss_task = ce(out.task_logits, train_tensors["task"][idx])
            loss_mod = ce_mod(out.modality_logits, train_tensors["modality"][idx])

            binary_targets = train_tensors["binary"][idx]
            loss_bin = (
                bce(out.temporal_logit, binary_targets[:, 0])
                + bce(out.spatial_logit, binary_targets[:, 1])
                + bce(out.language_logit, binary_targets[:, 2])
            ) / len(BINARY_HEADS)

            loss = (
                task_loss_weight * loss_task
                + modality_loss_weight * loss_mod
                + binary_loss_weight * loss_bin
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            batches += 1

        scheduler.step()

        val_metrics = evaluate_split(model, val_ex, val_x, "val", device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": epoch_loss / max(1, batches),
                "val_task_accuracy": val_metrics.task_accuracy,
                "val_combined_accuracy": val_metrics.combined_accuracy,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if val_metrics.task_accuracy > best_val:
            best_val = val_metrics.task_accuracy
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"  epoch {epoch + 1:>3}/{epochs}  "
                f"loss {epoch_loss / max(1, batches):.4f}  "
                f"val_task {val_metrics.task_accuracy:.3f}  "
                f"val_combined {val_metrics.combined_accuracy:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # -- final evaluation --------------------------------------------------
    train_metrics = evaluate_split(model, train_ex, train_x, "train", device)
    val_metrics = evaluate_split(model, val_ex, val_x, "val", device)
    test_metrics = evaluate_split(model, test_ex, test_x, "test", device)

    duration = time.time() - started

    # -- artifact ----------------------------------------------------------
    if artifact_dir is None:
        artifact_dir = Path("artifacts/router/router_adapter_v001")
    artifact_path = Path(artifact_dir)

    metadata: dict[str, Any] = {
        "artifact": "router_adapter",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config_hash": config_hash,
        "encoder": encoder_meta,
        "encoder_type": encoder_type,
        "adapter": {
            "num_parameters": model.num_parameters(),
            **model.config_dict(),
        },
        "corpus": {
            "total": len(corpus),
            "groups": len(corpus.groups()),
            "by_source": corpus.source_counts(),
            "by_task": corpus.task_counts(),
            "positives": {h: corpus.positives(h) for h in BINARY_HEADS},
            "hash": corpus_hash,
        },
        "split": report,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "task_loss_weight": task_loss_weight,
            "modality_loss_weight": modality_loss_weight,
            "binary_loss_weight": binary_loss_weight,
        },
        "metrics": {
            "train": train_metrics.to_dict(),
            "val": val_metrics.to_dict(),
            "test": test_metrics.to_dict(),
        },
        "history": history,
        "duration_seconds": round(duration, 2),
    }

    save_adapter(artifact_path, model, metadata)

    return TrainingResult(
        artifact_dir=artifact_path,
        metadata=metadata,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        split_report=report,
        corpus_hash=corpus_hash,
        duration_seconds=duration,
        history=history,
    )


# ---------------------------------------------------------------------------
# Stub embeddings (tests only)
# ---------------------------------------------------------------------------


def _stub_embeddings(corpus: RouterCorpus, dim: int = 384) -> np.ndarray:
    """Deterministic bag-of-hashes pseudo-embedding.

    Exists so `train_router` can be exercised in unit tests without downloading
    a 90 MB model. It is NOT a semantic encoder: two paraphrases get unrelated
    vectors. Any artifact trained with it is marked `encoder_type: "stub"` and
    must never be deployed.
    """
    out = np.zeros((len(corpus), dim), dtype=np.float32)
    for row, example in enumerate(corpus):
        for token in example.text.lower().split():
            h = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
            out[row, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


__all__ = [
    "train_router",
    "evaluate_split",
    "embed_corpus_cached",
    "SplitMetrics",
    "TrainingResult",
    "CORPUS_CACHE_VERSION",
]