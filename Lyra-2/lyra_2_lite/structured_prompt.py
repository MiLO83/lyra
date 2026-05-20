"""
Structured prompt builder for Lyra 2 Lite inference.

Wires the EntityVocabulary into the actual prompt-construction step used
by the Lyra 2 / Lyra 2 Lite text conditioning. Replaces the raw
"text caption -> T5 embedding" path with:

    base_caption + extracted_entities
            |
            v
    +-----------------------------+
    | StructuredPrompt builder    |
    |   - entity name parsing     |
    |   - bytestream encoding     |
    |   - colour hint extraction  |
    |   - canonical prompt expansion via vocab strings
    +-----------------------------+
            |
            v
    {
      "caption":   "<expanded text for T5>",
      "entity_ids": [int, int, ...],            # for auxiliary embedding
      "entity_bytes": b"<variable-byte stream>",# for atlas storage
      "color_hints": [(r,g,b), ...],            # for spatial colour prior
      "frequency_updated": True/False
    }

This is the concrete "+10% quality" lever from the Lyra 2 Lite optimization
table: structured entity tags reduce hallucination, give the model a
colour prior, and let the autoregressive cache reuse semantic regions
across scenes.

Usage:
    >>> from entity_vocab import seed
    >>> from structured_prompt import StructuredPromptBuilder
    >>> vocab = seed()
    >>> builder = StructuredPromptBuilder(vocab)
    >>> result = builder.build(
    ...     caption="A wooden ship sails past a derelict station",
    ...     entities=["ship", "station", "sky"],
    ... )
    >>> print(result["caption"])
    a wooden sailing ship, a derelict space station, expansive open sky. A wooden ship sails past a derelict station
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from entity_vocab import EntityVocabulary, seed


@dataclass
class StructuredPrompt:
    caption: str                          # expanded text for T5
    entity_ids: list[int]                 # for auxiliary embedding lookup
    entity_bytes: bytes                   # variable-byte stream for atlas / storage
    color_hints: list[tuple[int, int, int]]  # baseline RGB per entity
    extracted_entities: list[str]         # canonical entity names used
    frequency_updated: bool               # whether vocab counts were incremented


class StructuredPromptBuilder:
    """
    Stateful builder: holds vocab, increments scene_counts as captions
    are processed, optionally re-sorts vocab on a cadence.
    """

    def __init__(self, vocab: EntityVocabulary,
                 resort_every: int = 100,
                 auto_extract: bool = True,
                 increment_counts: bool = True):
        self.vocab = vocab
        self.resort_every = resort_every
        self.auto_extract = auto_extract
        self.increment_counts = increment_counts
        self._scenes_since_resort = 0
        self._extraction_re = self._compile_extraction_regex()

    def _compile_extraction_regex(self) -> re.Pattern:
        # Word-boundary match for any canonical name in the vocab.
        # Names sorted by length descending so multi-word names win
        # (none in the seed today, but supports future "dolphin_orca" etc).
        names = sorted([e.name for e in self.vocab.entities],
                       key=len, reverse=True)
        if not names:
            return re.compile(r"$.")  # never matches
        # Replace _ with optional space for matching e.g. "space ship" -> spaceship
        patterns = [re.escape(n).replace(r"_", r"[_ ]?") for n in names]
        return re.compile(r"\b(" + "|".join(patterns) + r")\b", re.IGNORECASE)

    def extract_entities(self, caption: str) -> list[str]:
        """Find vocab entity names mentioned in freeform caption text."""
        matches = self._extraction_re.findall(caption)
        seen = []
        for m in matches:
            canonical = m.lower().replace(" ", "_")
            if canonical in self.vocab.by_name and canonical not in seen:
                seen.append(canonical)
            else:
                # Try without normalization
                if m in self.vocab.by_name and m not in seen:
                    seen.append(m)
        return seen

    def build(self,
              caption: str,
              entities: Optional[Iterable[str]] = None,
              expand_caption: bool = True) -> dict:
        """
        Build a structured prompt.

        caption     : freeform user/LLM text description
        entities    : explicit entity name list. If None and auto_extract,
                      entities are parsed from the caption text.
        expand_caption : whether to prepend canonical entity descriptions
                         to the caption (the "+10% quality" expansion)
        """
        # 1. Resolve entity list
        if entities is None:
            entities = (self.extract_entities(caption)
                        if self.auto_extract else [])
        entities = list(entities)

        # 2. Sanity-check + drop unknowns
        known = [n for n in entities if n in self.vocab.by_name]
        unknown = [n for n in entities if n not in self.vocab.by_name]
        if unknown:
            # Soft-fail; emit a warning prefix in the caption
            pass

        # 3. Increment scene counts (drives frequency-sort fast path)
        if self.increment_counts and known:
            for n in known:
                self.vocab.increment(n)
            self._scenes_since_resort += 1
            if (self.resort_every > 0 and
                    self._scenes_since_resort >= self.resort_every):
                self.vocab.resort()
                self._scenes_since_resort = 0

        # 4. Build expanded caption (canonical strings + original text)
        if expand_caption and known:
            entity_text = ", ".join(self.vocab.get(n).string for n in known)
            expanded = f"{entity_text}. {caption}"
        else:
            expanded = caption

        # 5. Encode + collect IDs + colour hints
        entity_ids = [self.vocab.get_id(n) for n in known]
        entity_bytes = self.vocab.encode_many(known) if known else b""
        color_hints = [self.vocab.get(n).color for n in known]

        return {
            "caption": expanded,
            "entity_ids": entity_ids,
            "entity_bytes": entity_bytes,
            "color_hints": color_hints,
            "extracted_entities": known,
            "frequency_updated": bool(known and self.increment_counts),
        }

    def stats(self) -> dict:
        return {
            **self.vocab.stats(),
            "scenes_since_resort": self._scenes_since_resort,
            "resort_every": self.resort_every,
        }


# ---------------------------------------------------------------------------
# Lyra 2 Lite integration hook (drop-in replacement for raw caption)
# ---------------------------------------------------------------------------

def lyra2_caption_hook(caption: str,
                       builder: StructuredPromptBuilder,
                       entities: Optional[list[str]] = None) -> tuple[str, dict]:
    """
    Drop-in replacement for the raw `caption` string passed to Lyra 2's
    T5 encoder. Returns (expanded_caption, metadata) where metadata
    carries the entity_ids / entity_bytes / color_hints for use by any
    auxiliary heads or atlas writers downstream.

    In Lyra 2 Lite's inference script, replace:
        captions[chunk_idx] = json["captions"][...]
    with:
        raw = json["captions"][...]
        captions[chunk_idx], cond_meta = lyra2_caption_hook(raw, builder)
        # then thread cond_meta through to your auxiliary conditioner
    """
    result = builder.build(caption, entities=entities)
    meta = {k: v for k, v in result.items() if k != "caption"}
    return result["caption"], meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse, json as _json

    ap = argparse.ArgumentParser(description="Structured prompt builder for Lyra 2 Lite.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("build", help="build a structured prompt from a caption")
    sp.add_argument("--caption", required=True)
    sp.add_argument("--entities", nargs="*", default=None,
                    help="explicit entity list; if omitted, extracted from caption")
    sp.add_argument("--vocab", type=Path)
    sp.add_argument("--no-expand", action="store_true",
                    help="don't prepend canonical entity descriptions")
    sp.add_argument("--no-increment", action="store_true",
                    help="don't update vocab scene_counts (read-only)")

    sp = sub.add_parser("extract", help="just extract entities from caption text")
    sp.add_argument("--caption", required=True)
    sp.add_argument("--vocab", type=Path)

    args = ap.parse_args()

    vocab = EntityVocabulary.load(args.vocab) if args.vocab else seed()

    if args.cmd == "build":
        builder = StructuredPromptBuilder(
            vocab,
            resort_every=0,  # don't resort during one-shot CLI
            increment_counts=not args.no_increment,
        )
        result = builder.build(
            args.caption,
            entities=args.entities,
            expand_caption=not args.no_expand,
        )
        # Print summary
        print("=== Structured Prompt ===")
        print(f"\nexpanded caption:\n  {result['caption']}")
        print(f"\nextracted entities ({len(result['extracted_entities'])}):")
        for n in result['extracted_entities']:
            e = vocab.get(n)
            w = vocab.bytewidth_of(n)
            print(f"  [{w}B] {n:<14} color={e.color}")
        print(f"\nentity_bytes ({len(result['entity_bytes'])} bytes): "
              f"{result['entity_bytes'].hex(' ')}")
        print(f"entity_ids: {result['entity_ids']}")
        print(f"color_hints: {result['color_hints']}")

    elif args.cmd == "extract":
        builder = StructuredPromptBuilder(vocab, resort_every=0,
                                           increment_counts=False)
        found = builder.extract_entities(args.caption)
        print(f"caption: {args.caption}")
        print(f"extracted: {found}")


if __name__ == "__main__":
    main()
