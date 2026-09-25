"""SatQuery AI — router training corpus.

Two sources, deliberately:

1. A hand-written CURATED set. Short, natural, exactly the phrasings a user
   types. This is the part that has to be right; everything else is volume.

2. Template-generated examples from a shared template table. Each (task,
   template) pair carries a GROUP id.

The group tag is the important part (finding F4-3). `"show me the water body"`
and `"show me the road"` come from the same template and differ by one token.
Splitting them across train/val makes validation trivially easy and gives a
fake accuracy number. Splitting by group keeps every template on exactly one
side — the router-level analogue of scene-level splitting, and the same class
of bug Gate 1 exists to catch.

Hard negatives are first-class here, not an afterthought. The pairs that
matter:

    "describe the water body"    -> caption   (describe = language)
    "show me the water body"     -> grounding (show   = spatial)

    "what changed"               -> change, spatial_output=False
    "where did the change happen"-> change, spatial_output=True

    "compare these two images"   -> change    (two images, temporal)
    "compare optical and radar"  -> optical_sar (two modalities)

Every one of those is a single-token difference in the input and a different
label on the output. A router that has not seen such pairs will fail them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from router.label_space import (
    BINARY_HEADS,
    NUM_TASKS,
    TASK_CLASSES,
    TASK_TO_INDEX,
    MODALITY_CLASSES,
    MODALITY_TO_INDEX,
)

# ---------------------------------------------------------------------------
# Leakage boundaries
# ---------------------------------------------------------------------------

#: Groups whose name starts with this are hard-negative FAMILIES: sets of
#: deliberately confusable queries that must travel together through any split.
#: `hn_desc_vs_show` holds both "Describe the water body." (caption) and
#: "Show me the water body." (grounding) — splitting those apart would remove
#: the only signal that teaches the router the difference.
HARD_NEGATIVE_PREFIX = "hn_"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


def _group_index(corpus: "RouterCorpus") -> dict[str, list["RouterExample"]]:
    """Index a corpus by group id. One pass, used by validate() and the splitter."""
    index: dict[str, list[RouterExample]] = {}
    for example in corpus.examples:
        index.setdefault(example.group, []).append(example)
    return index


def _dominant_task(examples: list["RouterExample"]) -> str:
    """Most common task in a group, ties broken deterministically.

    Used only to choose which stratum a group belongs to when balancing splits.
    Hard-negative families deliberately span tasks, so they need a stratum too;
    the alphabetically-first task is as good a tie-break as any.
    """
    counts: dict[str, int] = {}
    for e in examples:
        counts[e.task] = counts.get(e.task, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


@dataclass
class RouterExample:
    """One labelled query.

    `group` is the leakage boundary: all examples sharing a group must land in
    the same split. `source` distinguishes hand-written from generated material
    so the two can be reported separately.
    """

    text: str
    task: str
    modality: str = "unknown"
    temporal: bool = False
    spatial_output: bool = False
    language_output: bool = True
    group: str = ""
    source: str = "template"

    def __post_init__(self) -> None:
        if self.task not in TASK_TO_INDEX:
            raise ValueError(f"unknown task label: {self.task!r}")
        if self.modality not in MODALITY_TO_INDEX:
            raise ValueError(f"unknown modality label: {self.modality!r}")
        if not self.text.strip():
            raise ValueError("example text must not be empty")
        if not self.group:
            raise ValueError(f"example {self.text!r} has no group tag")

    def label_tuple(self) -> tuple[int, int, tuple[float, ...]]:
        """(task_index, modality_index, (temporal, spatial, language)) as floats."""
        binaries = tuple(
            1.0 if getattr(self, head) else 0.0 for head in BINARY_HEADS
        )
        return TASK_TO_INDEX[self.task], MODALITY_TO_INDEX[self.modality], binaries

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "task": self.task,
            "modality": self.modality,
            "temporal": self.temporal,
            "spatial_output": self.spatial_output,
            "language_output": self.language_output,
            "group": self.group,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Curated examples — natural phrasing, hand-checked labels
# ---------------------------------------------------------------------------

CURATED: tuple[RouterExample, ...] = (
    # ---- caption -------------------------------------------------------
    RouterExample("Describe this image.", "caption", "unknown", False, False, True, "cur_cap", "curated"),
    RouterExample("Describe this satellite image.", "caption", "optical", False, False, True, "cur_cap", "curated"),
    RouterExample("Give me a description of the scene.", "caption", "unknown", False, False, True, "cur_cap", "curated"),
    RouterExample("What do you see in this image?", "caption", "unknown", False, False, True, "cur_cap", "curated"),
    RouterExample("Write a caption for this image.", "caption", "unknown", False, False, True, "cur_cap", "curated"),
    RouterExample("Summarize what is visible here.", "caption", "unknown", False, False, True, "cur_cap", "curated"),
    RouterExample("Describe the land cover in this scene.", "caption", "optical", False, False, True, "cur_cap", "curated"),
    RouterExample("Tell me about this image in detail.", "caption", "unknown", False, False, True, "cur_cap", "curated"),

    # ---- vqa -----------------------------------------------------------
    RouterExample("What land cover is visible?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),
    RouterExample("How many buildings are in this image?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),
    RouterExample("Is there water in this scene?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),
    RouterExample("What is the dominant land cover?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),
    RouterExample("Are there any roads visible?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),
    RouterExample("What type of vegetation is present?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),
    RouterExample("Is this an urban or rural area?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),
    RouterExample("Does this image contain a river?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),
    RouterExample("How many vehicles can you count?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),
    RouterExample("What is the weather like in this image?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),
    RouterExample("Is this image from an agricultural area?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),
    RouterExample("What season does this image show?", "vqa", "optical", False, False, True, "cur_vqa", "curated"),

    # ---- grounding -----------------------------------------------------
    RouterExample("Show me the water body.", "grounding", "optical", False, True, True, "cur_grd", "curated"),
    RouterExample("Locate the buildings.", "grounding", "optical", False, True, True, "cur_grd", "curated"),
    RouterExample("Where is the road?", "grounding", "optical", False, True, True, "cur_grd", "curated"),
    RouterExample("Highlight the vegetation.", "grounding", "optical", False, True, True, "cur_grd", "curated"),
    RouterExample("Find the bridge in this image.", "grounding", "optical", False, True, True, "cur_grd", "curated"),
    RouterExample("Point out the harbor.", "grounding", "optical", False, True, True, "cur_grd", "curated"),
    RouterExample("Draw a box around the river.", "grounding", "optical", False, True, True, "cur_grd", "curated"),
    RouterExample("Can you show me where the water body is?", "grounding", "optical", False, True, True, "cur_grd", "curated"),
    RouterExample("Mark the location of the airport.", "grounding", "optical", False, True, True, "cur_grd", "curated"),
    RouterExample("Where are the trees located?", "grounding", "optical", False, True, True, "cur_grd", "curated"),

    # ---- change --------------------------------------------------------
    RouterExample("What changed between these images?", "change", "optical", True, False, True, "cur_chg", "curated"),
    RouterExample("What changed?", "change", "optical", True, False, True, "cur_chg", "curated"),
    RouterExample("Has the urban area increased?", "change", "optical", True, False, True, "cur_chg", "curated"),
    RouterExample("Compare the before and after images.", "change", "optical", True, False, True, "cur_chg", "curated"),
    RouterExample("Describe the changes.", "change", "optical", True, False, True, "cur_chg", "curated"),
    RouterExample("Did any buildings appear?", "change", "optical", True, False, True, "cur_chg", "curated"),
    RouterExample("Where did the change happen?", "change", "optical", True, True, True, "cur_chg", "curated"),
    RouterExample("Show me the changed regions.", "change", "optical", True, True, True, "cur_chg", "curated"),
    RouterExample("What changed and show me where?", "change", "optical", True, True, True, "cur_chg", "curated"),
    RouterExample("Highlight the areas that changed.", "change", "optical", True, True, True, "cur_chg", "curated"),
    RouterExample("Locate where new construction appeared.", "change", "optical", True, True, True, "cur_chg", "curated"),
    RouterExample("How much forest was lost between the two dates?", "change", "optical", True, False, True, "cur_chg", "curated"),

    # ---- optical_sar ---------------------------------------------------
    RouterExample("Compare the optical and radar images.", "optical_sar", "optical_sar", False, False, True, "cur_osr", "curated"),
    RouterExample("Compare optical and SAR to identify built-up regions.", "optical_sar", "optical_sar", False, False, True, "cur_osr", "curated"),
    RouterExample("Use both the radar and optical images.", "optical_sar", "optical_sar", False, False, True, "cur_osr", "curated"),
    RouterExample("Jointly analyse the optical and SAR pair.", "optical_sar", "optical_sar", False, False, True, "cur_osr", "curated"),
    RouterExample("What can the radar tell us that the optical cannot?", "optical_sar", "optical_sar", False, False, True, "cur_osr", "curated"),
    RouterExample("Fuse the optical and SAR data.", "optical_sar", "optical_sar", False, False, True, "cur_osr", "curated"),
    RouterExample("Analyse the co-registered optical and radar pair.", "optical_sar", "optical_sar", False, False, True, "cur_osr", "curated"),
    RouterExample("Show built-up areas using radar and optical together.", "optical_sar", "optical_sar", False, False, True, "cur_osr", "curated"),

    # ---- unsupported ---------------------------------------------------
    RouterExample("Book me a flight to Delhi.", "unsupported", "unknown", False, False, False, "cur_uns", "curated"),
    RouterExample("What is the weather tomorrow?", "unsupported", "unknown", False, False, False, "cur_uns", "curated"),
    RouterExample("Write me a poem about satellites.", "unsupported", "unknown", False, False, False, "cur_uns", "curated"),
    RouterExample("Who won the cricket match?", "unsupported", "unknown", False, False, False, "cur_uns", "curated"),
    RouterExample("Send an email to my supervisor.", "unsupported", "unknown", False, False, False, "cur_uns", "curated"),
    RouterExample("What is the capital of France?", "unsupported", "unknown", False, False, False, "cur_uns", "curated"),

    # ---- explicit hard negatives ---------------------------------------
    # One-token difference, different label. These are the tests that matter.
    RouterExample("Describe the water body.", "caption", "optical", False, False, True, "hn_desc_vs_show", "curated"),
    RouterExample("Show me the water body.", "grounding", "optical", False, True, True, "hn_desc_vs_show", "curated"),
    RouterExample("Describe the road.", "caption", "optical", False, False, True, "hn_desc_vs_show", "curated"),
    RouterExample("Show me the road.", "grounding", "optical", False, True, True, "hn_desc_vs_show", "curated"),
    RouterExample("Describe the buildings.", "caption", "optical", False, False, True, "hn_desc_vs_show", "curated"),
    RouterExample("Show me the buildings.", "grounding", "optical", False, True, True, "hn_desc_vs_show", "curated"),

    RouterExample("What changed?", "change", "optical", True, False, True, "hn_what_vs_where", "curated"),
    RouterExample("Where did the change happen?", "change", "optical", True, True, True, "hn_what_vs_where", "curated"),
    RouterExample("Describe the changes.", "change", "optical", True, False, True, "hn_what_vs_where", "curated"),
    RouterExample("Show me where the changes are.", "change", "optical", True, True, True, "hn_what_vs_where", "curated"),

    RouterExample("Compare these two images.", "change", "optical", True, False, True, "hn_temporal_vs_modality", "curated"),
    RouterExample("Compare optical and radar.", "optical_sar", "optical_sar", False, False, True, "hn_temporal_vs_modality", "curated"),
    RouterExample("What is different between these two dates?", "change", "optical", True, False, True, "hn_temporal_vs_modality", "curated"),
    RouterExample("What is different between the optical and SAR views?", "optical_sar", "optical_sar", False, False, True, "hn_temporal_vs_modality", "curated"),
)


# ---------------------------------------------------------------------------
# Templates — volume, group-tagged
# ---------------------------------------------------------------------------

#: (group_id, task, template, modality, temporal, spatial, language)
#: `{}` is replaced by a subject from the matching subject list.
TEMPLATES: tuple[tuple[str, str, str, str, bool, bool, bool], ...] = (
    # caption
    ("t_cap_a", "caption", "Describe the {}.", "optical", False, False, True),
    ("t_cap_b", "caption", "Give a detailed description of the {}.", "optical", False, False, True),
    ("t_cap_c", "caption", "What does the {} look like?", "optical", False, False, True),
    ("t_cap_d", "caption", "Write a caption describing the {}.", "optical", False, False, True),
    ("t_cap_e", "caption", "Summarize the {} visible in this image.", "optical", False, False, True),

    # vqa
    ("t_vqa_a", "vqa", "Is the {} present in this image?", "optical", False, False, True),
    ("t_vqa_b", "vqa", "How many {} are visible?", "optical", False, False, True),
    ("t_vqa_c", "vqa", "What kind of {} is shown here?", "optical", False, False, True),
    ("t_vqa_d", "vqa", "Can you tell if this contains {}?", "optical", False, False, True),
    ("t_vqa_e", "vqa", "Are there any {} in the scene?", "optical", False, False, True),

    # grounding
    ("t_grd_a", "grounding", "Show me the {}.", "optical", False, True, True),
    ("t_grd_b", "grounding", "Locate the {}.", "optical", False, True, True),
    ("t_grd_c", "grounding", "Where is the {}?", "optical", False, True, True),
    ("t_grd_d", "grounding", "Highlight the {}.", "optical", False, True, True),
    ("t_grd_e", "grounding", "Draw a box around the {}.", "optical", False, True, True),
    ("t_grd_f", "grounding", "Point out the {} on the map.", "optical", False, True, True),

    # change
    ("t_chg_a", "change", "What happened to the {} between the two images?", "optical", True, False, True),
    ("t_chg_b", "change", "Has the {} changed?", "optical", True, False, True),
    ("t_chg_c", "change", "Describe how the {} changed over time.", "optical", True, False, True),
    ("t_chg_d", "change", "Where did the {} change?", "optical", True, True, True),
    ("t_chg_e", "change", "Show the regions where the {} changed.", "optical", True, True, True),
    ("t_chg_f", "change", "How much did the {} increase or decrease?", "optical", True, False, True),

    # optical_sar
    ("t_osr_a", "optical_sar", "Compare the optical and radar views of the {}.", "optical_sar", False, False, True),
    ("t_osr_b", "optical_sar", "Use both sensors to identify the {}.", "optical_sar", False, False, True),
    ("t_osr_c", "optical_sar", "How does the {} appear differently in SAR and optical?", "optical_sar", False, False, True),
    ("t_osr_d", "optical_sar", "Analyse the {} using the co-registered optical and SAR pair.", "optical_sar", False, False, True),

    # unsupported
    #
    # Volume matters here more than anywhere else. Group-level splitting keeps
    # every template on one side of the boundary, so a class with only four
    # templates ends up with two of them available for training — and a template
    # the model never saw cannot be learned, only guessed at. With four templates
    # the "Recommend a restaurant near X" phrasing fell entirely into test and
    # the router, having never seen it, read "near <scene noun>" as a grounding
    # request. That is correct behaviour on insufficient data, so the fix is more
    # distinct phrasings, not a looser gate.
    ("t_uns_a", "unsupported", "Tell me a joke about {}.", "unknown", False, False, False),
    ("t_uns_b", "unsupported", "What is the population of {}?", "unknown", False, False, False),
    ("t_uns_c", "unsupported", "Recommend a restaurant near {}.", "unknown", False, False, False),
    ("t_uns_d", "unsupported", "Book me a flight to {}.", "unknown", False, False, False),
    ("t_uns_e", "unsupported", "Who won the match in {}?", "unknown", False, False, False),
    ("t_uns_f", "unsupported", "Translate this sentence into {}.", "unknown", False, False, False),
    ("t_uns_g", "unsupported", "Write a poem about {}.", "unknown", False, False, False),
    ("t_uns_h", "unsupported", "What is the weather in {} tomorrow?", "unknown", False, False, False),
    ("t_uns_i", "unsupported", "Send an email about {}.", "unknown", False, False, False),
    ("t_uns_j", "unsupported", "Summarize this news article about {}.", "unknown", False, False, False),

    # change — additional distinct phrasings so the class survives group-level
    # splitting with several templates still available for training.
    ("t_chg_g", "change", "Compare the two acquisitions of the {}.", "optical", True, False, True),
    ("t_chg_h", "change", "What is different about the {} between the two dates?", "optical", True, False, True),
    ("t_chg_i", "change", "Did the {} expand or shrink?", "optical", True, False, True),
    ("t_chg_j", "change", "Point out where the {} was modified.", "optical", True, True, True),

    # caption — additional phrasings; caption is a mandatory capability and
    # needs enough templates to be trainable and measurable at once.
    ("t_cap_f", "caption", "Provide an overview of the {}.", "optical", False, False, True),
    ("t_cap_g", "caption", "Explain what is happening in the {}.", "optical", False, False, True),
    ("t_cap_h", "caption", "Give me a short report on the {}.", "optical", False, False, True),

    # grounding — additional phrasings.
    ("t_grd_g", "grounding", "Give me the coordinates of the {}.", "optical", False, True, True),
    # NOTE: an earlier version had "Which part of the image contains the {}?"
    # here. It was removed because its SURFACE FORM is a VQA frame — the same
    # interrogative ("which") and the same verb ("contains") as t_vqa_d
    # "Can you tell if this contains {}?". Labelling that pattern `grounding`
    # is label noise, not a hard negative, and the model dutifully learned the
    # contradiction: every one of the 15 grounding errors in test was that
    # single template, predicted `vqa` at 0.78-0.98 confidence while the other
    # seven grounding templates scored perfectly. Spatial intent needs spatial
    # phrasing; ambiguity belongs in the `hn_*` families where it is deliberate.
    ("t_grd_h", "grounding", "Trace the outline of the {}.", "optical", False, True, True),
)

#: Subject vocabulary per task family.
SUBJECTS: dict[str, tuple[str, ...]] = {
    "caption": (
        "scene", "image", "landscape", "area", "region",
        "terrain", "land cover", "cityscape", "coastline", "farmland",
    ),
    "vqa": (
        "water body", "road", "building", "forest", "river",
        "bridge", "harbor", "vehicle", "tree", "field",
        "airport", "railway", "dam", "lake", "industrial area",
    ),
    "grounding": (
        "water body", "building", "road", "river", "forest",
        "bridge", "harbor", "lake", "airport", "dam",
        "railway line", "parking lot", "swimming pool", "stadium", "quarry",
    ),
    "change": (
        "urban area", "forest", "water body", "built-up area", "vegetation",
        "farmland", "road network", "construction site", "coastline", "wetland",
    ),
    "optical_sar": (
        "built-up area", "water body", "forest", "agricultural field", "urban region",
        "flooded area", "road network", "industrial zone", "coastline", "wetland",
    ),
    # Unsupported subjects deliberately mix two kinds:
    #
    #   * satellite-imagery nouns used in a NON-imagery frame (the moon, the
    #     ocean, the ISS) — the valuable half. "Recommend a restaurant near the
    #     ocean" contains a scene noun but is not an imagery request, so it
    #     teaches that the TEMPLATE dominates the subject.
    #   * unambiguous out-of-domain entities (Delhi, cricket, French)
    #
    # Held to ten. Ten templates x ten subjects keeps the class at ~14% of
    # training; twenty subjects pushed it to 30% and the inverse-frequency
    # class weight (0.548) had to work hard to compensate. Template COUNT is
    # what buys group-level trainability; subject count only buys volume.
    "unsupported": (
        "the moon", "the ocean", "galaxies", "the ISS", "satellites",
        "Delhi", "Mumbai", "cricket", "French", "the Himalayas",
    ),
}


def _generate_templates() -> list[RouterExample]:
    """Expand the template table into examples, one group per template."""
    out: list[RouterExample] = []
    for group, task, template, modality, temporal, spatial, language in TEMPLATES:
        subjects = SUBJECTS.get(task, SUBJECTS["caption"])
        for subject in subjects:
            out.append(
                RouterExample(
                    text=template.format(subject),
                    task=task,
                    modality=modality,
                    temporal=temporal,
                    spatial_output=spatial,
                    language_output=language,
                    group=group,
                    source="template",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass
class RouterCorpus:
    """A labelled, group-tagged corpus ready for splitting and caching."""

    examples: list[RouterExample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def texts(self) -> list[str]:
        return [e.text for e in self.examples]

    def groups(self) -> set[str]:
        return {e.group for e in self.examples}

    def task_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in TASK_CLASSES}
        for e in self.examples:
            counts[e.task] += 1
        return counts

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.examples:
            counts[e.source] = counts.get(e.source, 0) + 1
        return counts

    def positives(self, head: str) -> int:
        return sum(1 for e in self.examples if getattr(e, head))

    def dedupe(self) -> tuple["RouterCorpus", list[str]]:
        """Remove exact duplicate texts, keeping the first occurrence.

        A query appearing under two different labels is a labelling conflict and
        is reported rather than silently resolved.
        """
        seen: dict[str, str] = {}
        kept: list[RouterExample] = []
        conflicts: list[str] = []

        for example in self.examples:
            key = example.text.strip().lower()
            if key in seen:
                if seen[key] != example.task:
                    conflicts.append(
                        f"{example.text!r}: {seen[key]} vs {example.task}"
                    )
                continue
            seen[key] = example.task
            kept.append(example)

        return RouterCorpus(kept), conflicts

    def validate(self) -> list[str]:
        """Structural sanity checks. Returns a list of problems (empty = good)."""
        problems: list[str] = []

        if not self.examples:
            problems.append("corpus is empty")
            return problems

        # Group/task consistency.
        #
        # A group is a LEAKAGE boundary, not a label constraint: every example
        # in a group goes to the same split, whatever its label. That is exactly
        # what the hard-negative families need — `hn_desc_vs_show` contains both
        # "Describe the water body." (caption) and "Show me the water body."
        # (grounding), and splitting those two apart would destroy the whole
        # point of the pair.
        #
        # So a group MAY span tasks. What must never happen is two examples with
        # the same TEXT carrying different labels; that is a labelling conflict
        # and `dedupe()` reports it.
        spanning = {
            group: sorted({e.task for e in examples})
            for group, examples in _group_index(self).items()
            if len({e.task for e in examples}) > 1
        }
        for group, tasks in spanning.items():
            if not group.startswith(HARD_NEGATIVE_PREFIX):
                problems.append(
                    f"group {group!r} spans multiple tasks {tasks} but is not a "
                    f"hard-negative family (expected prefix "
                    f"{HARD_NEGATIVE_PREFIX!r}); either split it or rename it"
                )

        # Every task class must be represented, or the head cannot learn it.
        counts = self.task_counts()
        for name in TASK_CLASSES:
            if counts[name] == 0:
                problems.append(f"task class {name!r} has no examples")

        # Unsupported queries must never claim a capability.
        for e in self.examples:
            if e.task == "unsupported" and (e.temporal or e.spatial_output or e.language_output):
                problems.append(
                    f"unsupported example {e.text!r} claims a capability"
                )

        # Temporal/spatial coherence.
        for e in self.examples:
            if e.task == "change" and not e.temporal:
                problems.append(f"change example {e.text!r} is not marked temporal")
            if e.task == "grounding" and not e.spatial_output:
                problems.append(
                    f"grounding example {e.text!r} is not marked spatial_output"
                )
            if e.task == "optical_sar" and e.modality != "optical_sar":
                problems.append(
                    f"optical_sar example {e.text!r} has modality {e.modality!r}"
                )

        return problems


def build_corpus(
    n_template_repeats: int = 1,
    seed: int = 42,
    include_curated: bool = True,
    include_templates: bool = True,
) -> RouterCorpus:
    """Assemble the full router corpus.

    Args:
        n_template_repeats: duplicate the template block N times with shuffled
            subject ordering. Volume without new phrasings; keep at 1 unless the
            adapter is clearly underfitting.
        seed: controls subject shuffling.
        include_curated / include_templates: ablation switches.
    """
    rng = random.Random(seed)
    examples: list[RouterExample] = []

    if include_curated:
        examples.extend(CURATED)

    if include_templates:
        for _ in range(max(1, n_template_repeats)):
            block = _generate_templates()
            rng.shuffle(block)
            examples.extend(block)

    return RouterCorpus(examples)


def split_by_group(
    corpus: RouterCorpus,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    seed: int = 42,
    hard_negatives_to_test: bool = True,
) -> dict[str, list[RouterExample]]:
    """Split by GROUP, never by example (finding F4-3).

    Mirrors `evaluation.leakage.assign_splits_by_scene` deliberately: the failure
    mode is identical, so the guard should look identical too. Groups are sorted
    before shuffling so the result does not depend on corpus order.

    Two guarantees beyond a naive ratio split:

    1. **Every task class reaches train.** A random group split can strand a
       class entirely in val/test, and then its head can never learn it. Groups
       are walked in shuffled order and any group that covers a not-yet-covered
       task is pulled into train first, before the ratio is topped up.

    2. **Hard-negative families are held out.** They are the cases the router
       most needs measured, so by default they go to test rather than training
       on the exact pairs being scored.

    Args:
        train_ratio: target fraction of non-held-out groups for training.
        val_ratio: target fraction for validation.
        seed: controls the shuffle. The same seed always yields the same split.
        hard_negatives_to_test: force `hn_*` groups into test.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1, got {train_ratio} + {val_ratio}"
        )

    by_group = _group_index(corpus)
    if not by_group:
        raise ValueError("cannot split an empty corpus")

    groups = sorted(by_group)

    # -- groups held out of training entirely ------------------------------
    forced_test = {
        g for g in groups
        if hard_negatives_to_test and g.startswith(HARD_NEGATIVE_PREFIX)
    }
    pool = [g for g in groups if g not in forced_test]
    if not pool:
        raise ValueError(
            "every group is a hard-negative family; there is nothing to train on"
        )

    # -- stratify by dominant task ----------------------------------------
    # Group-level splitting is what prevents leakage. It is NOT sufficient on
    # its own: a naive group split can strand an entire task class in one
    # partition. That is not hypothetical — with 38 groups and a flat 75/15
    # allocation, `vqa` (the most important mandatory task) and `unsupported`
    # both ended up with ZERO test examples, so the acceptance gate could not
    # measure them and their recall printed as a misleading 0.000.
    #
    # Stratifying by task fixes that WITHOUT weakening leakage safety: a group
    # still travels to exactly one split, whatever its labels. Only the choice
    # of which split changes.
    test_ratio = 1.0 - train_ratio - val_ratio

    strata: dict[str, list[str]] = {}
    for group in pool:
        strata.setdefault(_dominant_task(by_group[group]), []).append(group)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}

    for task in sorted(strata):
        group_list = list(strata[task])
        rng.shuffle(group_list)
        n = len(group_list)

        if n == 1:
            # Cannot be spread without losing the class from training. Training
            # must win: an unlearnable class is worse than an unmeasured one.
            assignment[group_list[0]] = "train"
            continue

        if n == 2:
            # One to train (learnable), one to test (measurable). Validation
            # coverage for this task comes from the corpus at large.
            assignment[group_list[0]] = "train"
            assignment[group_list[1]] = "test"
            continue

        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        # Keep at least one group for training, whatever the ratios ask for.
        while n_test + n_val >= n:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break

        n_train = n - n_test - n_val
        for i, group in enumerate(group_list):
            if i < n_train:
                assignment[group] = "train"
            elif i < n_train + n_val:
                assignment[group] = "val"
            else:
                assignment[group] = "test"

    # Hard-negative families are held out of training entirely, by design.
    for group in sorted(forced_test):
        assignment[group] = "test"

    # Every group must have been assigned exactly once.
    missing = [g for g in groups if g not in assignment]
    if missing:
        raise ValueError(f"split left {len(missing)} group(s) unassigned: {missing[:5]}")

    # Emit in sorted group order. The seed chooses *which* group goes where; it
    # must not control the order of the returned lists, or downstream code
    # taking records[:N] would silently depend on the seed.
    out: dict[str, list[RouterExample]] = {"train": [], "val": [], "test": []}
    for group in groups:
        out[assignment[group]].extend(by_group[group])
    return out


def split_leakage_report(split: dict[str, list[RouterExample]]) -> dict[str, object]:
    """Verify no group crosses a split boundary. Returns a report dict."""
    group_splits: dict[str, set[str]] = {}
    for name, examples in split.items():
        for e in examples:
            group_splits.setdefault(e.group, set()).add(name)

    offenders = {g: sorted(s) for g, s in group_splits.items() if len(s) > 1}

    return {
        "clean": not offenders,
        "groups_across_splits": offenders,
        "sizes": {name: len(examples) for name, examples in split.items()},
        "task_counts": {
            name: _count_tasks(examples) for name, examples in split.items()
        },
        "unique_groups": {name: len({e.group for e in examples})
                          for name, examples in split.items()},
    }


def _count_tasks(examples: Iterable[RouterExample]) -> dict[str, int]:
    counts = {name: 0 for name in TASK_CLASSES}
    for e in examples:
        counts[e.task] += 1
    return counts


__all__ = [
    "HARD_NEGATIVE_PREFIX",
    "RouterExample",
    "RouterCorpus",
    "CURATED",
    "TEMPLATES",
    "SUBJECTS",
    "build_corpus",
    "split_by_group",
    "split_leakage_report",
]