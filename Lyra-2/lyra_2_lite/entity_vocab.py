"""
SemanticEntity + EntityVocabulary for Lyra 2 Lite structured prompting.

Design (MiLO, 2026-05-20): sortable class with (scene_count, color, string)
where scene_count drives variable-byte priority encoding (1/2/3 bytes).
Self-tuning -- high-frequency entities automatically get the 1-byte fast path.

Variable-byte priority encoding (LEB128-style):

    byte 0 msb = 0 ->  1 byte total, IDs 0..127           (top 128 frequent)
    byte 0 msb = 1, byte 1 msb = 0 -> 2 bytes total, IDs 128..16511  (next 16k)
    byte 0,1 msb = 1, byte 2 msb = 0 -> 3 bytes total, IDs 16512..2113663

Same vocabulary used across:
  - caption parser:    text -> list of entity IDs
  - Lyra 2 Lite cond:  entity IDs -> learned embedding lookups
  - voxel atlas:       per-voxel semantic tags (variable-byte stream)
  - renderer:          entity ID -> color / material lookup
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class SemanticEntity:
    name: str
    string: str                # expanded prompt text
    color: tuple               # baseline RGB hint (0-255 ints)
    scene_count: int = 0       # cumulative scenes this has appeared in

    def __lt__(self, other):
        # Descending by scene_count; ties broken alphabetically by name.
        return (-self.scene_count, self.name) < (-other.scene_count, other.name)

    def to_dict(self):
        return {"name": self.name, "string": self.string,
                "color": list(self.color), "scene_count": self.scene_count}

    @classmethod
    def from_dict(cls, d):
        return cls(name=d["name"], string=d["string"],
                   color=tuple(d["color"]), scene_count=d.get("scene_count", 0))


class EntityVocabulary:
    """
    Self-organizing entity vocabulary with variable-byte priority encoding.

    Capacity: 128 (1-byte) + 16,384 (2-byte) + 2,097,152 (3-byte) = 2,113,664 entities.
    Average storage per tag for a typical project: ~1.3 bytes.
    """
    MAX_1BYTE = 128
    OFFSET_2BYTE = 128
    MAX_2BYTE_OFFSET = 16384
    MAX_2BYTE = OFFSET_2BYTE + MAX_2BYTE_OFFSET   # 16,512
    OFFSET_3BYTE = MAX_2BYTE
    MAX_3BYTE_OFFSET = 1 << 21
    CAPACITY = OFFSET_3BYTE + MAX_3BYTE_OFFSET    # 2,113,664

    def __init__(self):
        self.entities: list[SemanticEntity] = []
        self.by_name: dict[str, int] = {}

    def __len__(self):
        return len(self.entities)

    def __contains__(self, name: str):
        return name in self.by_name

    # ---- mutation ----

    def register(self, name: str, string: str, color, count: int = 0) -> int:
        if name in self.by_name:
            return self.by_name[name]
        if len(self.entities) >= self.CAPACITY:
            raise OverflowError(f"vocabulary at capacity ({self.CAPACITY})")
        self.entities.append(SemanticEntity(name, string, tuple(color), count))
        self.by_name[name] = len(self.entities) - 1
        return self.by_name[name]

    def increment(self, name: str, by: int = 1):
        self.entities[self.by_name[name]].scene_count += by

    def increment_many(self, names: Iterable[str]):
        for n in names:
            self.increment(n)

    # ---- lookup ----

    def get(self, name: str) -> SemanticEntity:
        return self.entities[self.by_name[name]]

    def get_id(self, name: str) -> int:
        return self.by_name[name]

    def get_by_id(self, eid: int) -> SemanticEntity:
        return self.entities[eid]

    # ---- variable-byte encoding ----

    @classmethod
    def _encode_id(cls, eid: int) -> bytes:
        if eid < cls.MAX_1BYTE:
            return bytes([eid])
        if eid < cls.MAX_2BYTE:
            adj = eid - cls.OFFSET_2BYTE
            return bytes([0x80 | (adj >> 7) & 0x7F, adj & 0x7F])
        adj = eid - cls.OFFSET_3BYTE
        if adj >= cls.MAX_3BYTE_OFFSET:
            raise OverflowError(f"id {eid} exceeds 3-byte capacity")
        return bytes([0x80 | (adj >> 14) & 0x7F,
                      0x80 | (adj >> 7) & 0x7F,
                      adj & 0x7F])

    @classmethod
    def _decode_id(cls, stream: bytes, pos: int = 0) -> tuple[int, int]:
        """Returns (entity_id, next_position)."""
        b0 = stream[pos]
        if b0 < 0x80:
            return (b0, pos + 1)
        b1 = stream[pos + 1]
        if b1 < 0x80:
            return (cls.OFFSET_2BYTE + (((b0 & 0x7F) << 7) | b1), pos + 2)
        b2 = stream[pos + 2]
        return (cls.OFFSET_3BYTE +
                (((b0 & 0x7F) << 14) | ((b1 & 0x7F) << 7) | b2),
                pos + 3)

    def encode(self, name: str) -> bytes:
        return self._encode_id(self.by_name[name])

    def encode_many(self, names: Iterable[str]) -> bytes:
        return b"".join(self._encode_id(self.by_name[n]) for n in names)

    def decode(self, stream: bytes) -> list[int]:
        out, pos = [], 0
        while pos < len(stream):
            eid, pos = self._decode_id(stream, pos)
            out.append(eid)
        return out

    def decode_names(self, stream: bytes) -> list[str]:
        return [self.entities[i].name for i in self.decode(stream)]

    # ---- resort + remap ----

    def resort(self) -> dict[int, int]:
        """
        Re-sort entities by frequency (then alphabetical).
        Returns {old_id: new_id} -- use remap_stream() to rewrite existing atlases.
        """
        old = {e.name: i for i, e in enumerate(self.entities)}
        self.entities.sort()
        new = {e.name: i for i, e in enumerate(self.entities)}
        self.by_name = new
        return {old[n]: new[n] for n in old}

    def remap_stream(self, stream: bytes, remap: dict[int, int]) -> bytes:
        ids = self.decode(stream)
        new_ids = [remap.get(i, i) for i in ids]
        return b"".join(self._encode_id(i) for i in new_ids)

    # ---- prompt building ----

    def prompt_for(self, names: Iterable[str], joiner: str = ", ") -> str:
        return joiner.join(self.get(n).string for n in names)

    def prompt_for_ids(self, ids: Iterable[int], joiner: str = ", ") -> str:
        return joiner.join(self.entities[i].string for i in ids)

    # ---- diagnostics ----

    def bytewidth_of(self, name_or_id) -> int:
        eid = name_or_id if isinstance(name_or_id, int) else self.by_name[name_or_id]
        if eid < self.MAX_1BYTE:
            return 1
        if eid < self.MAX_2BYTE:
            return 2
        return 3

    def stats(self) -> dict:
        n = len(self.entities)
        total_counts = sum(e.scene_count for e in self.entities)
        if total_counts > 0:
            avg = sum(self.bytewidth_of(i) * e.scene_count
                      for i, e in enumerate(self.entities)) / total_counts
        else:
            avg = float("nan")
        return {
            "entities": n,
            "1_byte_slots_used": min(n, self.MAX_1BYTE),
            "2_byte_slots_used": max(0, min(n, self.MAX_2BYTE) - self.MAX_1BYTE),
            "3_byte_slots_used": max(0, n - self.MAX_2BYTE),
            "capacity": self.CAPACITY,
            "scene_count_total": total_counts,
            "avg_bytes_per_tag": round(avg, 3),
        }

    # ---- persistence ----

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([e.to_dict() for e in self.entities], indent=2))

    @classmethod
    def load(cls, path) -> "EntityVocabulary":
        data = json.loads(Path(path).read_text())
        v = cls()
        for d in data:
            e = SemanticEntity.from_dict(d)
            v.entities.append(e)
            v.by_name[e.name] = len(v.entities) - 1
        return v


# ---------------------------------------------------------------------------
# Seed vocabulary (MiLOandOpUSPixels v2 entities + basics)
# ---------------------------------------------------------------------------

SEED_ENTITIES = [
    # Basics
    ("sky",       "expansive open sky",                                (135, 180, 220)),
    ("ground",    "rocky ground and bare earth",                       ( 90,  70,  50)),
    ("water",     "rippling water surface",                            ( 60, 110, 160)),
    ("character", "a human character",                                 (210, 180, 150)),

    # Creatures
    ("dolphin",   "a sleek dolphin with mottled grey skin",            ( 80, 130, 170)),
    ("whale",     "a massive blue whale",                              ( 60,  80, 110)),
    ("shark",     "a sleek prowling shark",                            ( 80,  90, 100)),
    ("fish",      "a school of bright tropical fish",                  (180, 120,  60)),
    ("jellyfish", "a translucent glowing jellyfish",                   (200, 220, 255)),
    ("octopus",   "a coiling octopus with patterned skin",             (180,  60,  80)),
    ("dragon",    "an immense scaled dragon",                          (140,  40,  40)),
    ("bird",      "a soaring bird",                                    (210, 200, 180)),
    ("wolf",      "a lone wolf with grey-black fur",                   ( 90,  80,  70)),
    ("robot",     "a humanoid robot with painted plating",             (160, 160, 170)),
    ("alien",     "an alien creature with translucent skin",           (130, 200, 130)),
    ("cyborg",    "a part-organic cyborg with exposed circuitry",      (120, 130, 140)),

    # Space
    ("star",      "a bright distant star",                             (255, 240, 200)),
    ("sun",       "a brilliant glaring sun",                           (255, 220, 120)),
    ("planet",    "a planet hanging in space",                         (140, 110,  80)),
    ("moon",      "a pale cratered moon",                              (200, 200, 200)),
    ("asteroid",  "a jagged rocky asteroid",                           (100,  90,  80)),
    ("comet",     "a comet with a luminous tail",                      (220, 240, 255)),
    ("nebula",    "a vibrant gaseous nebula",                          (180,  80, 200)),
    ("galaxy",    "a sweeping spiral galaxy",                          (140, 120, 200)),
    ("blackhole", "a black hole bending light around itself",          ( 10,  10,  20)),
    ("wormhole",  "a shimmering wormhole tunnel",                      ( 80, 120, 220)),

    # Structures
    ("station",   "a derelict space station",                          (150, 150, 160)),
    ("ship",      "a wooden sailing ship",                             (120,  80,  50)),
    ("spaceship", "a futuristic spaceship",                            (180, 190, 200)),
    ("satellite", "a small orbital satellite",                         (190, 190, 190)),
    ("building",  "an aged stone building",                            (160, 150, 130)),
    ("tower",     "a tall stone tower",                                (150, 140, 120)),
    ("pyramid",   "an ancient stepped pyramid",                        (190, 170, 130)),
    ("temple",    "a moss-covered jungle temple",                      (120, 130,  90)),
    ("castle",    "a sprawling stone castle",                          (140, 140, 130)),
    ("ruins",     "crumbling ancient ruins",                           (160, 150, 130)),
    ("bridge",    "an old wooden footbridge",                          (110,  85,  60)),
    ("gate",      "a heavy iron-bound gate",                           ( 90,  70,  50)),
    ("portal",    "a glowing magical portal",                          (160, 100, 240)),
    ("dome",      "a translucent geodesic dome",                       (200, 220, 230)),

    # Nature
    ("tree",      "a gnarled tropical tree",                           ( 80, 110,  60)),
    ("forest",    "a dense forest of tall trees",                      ( 60,  90,  50)),
    ("mountain",  "a snow-capped mountain",                            (160, 170, 180)),
    ("volcano",   "an active volcano with glowing magma",              (110,  60,  50)),
    ("island",    "a small tropical island",                           (180, 170, 130)),
    ("cave",      "a dark cave mouth",                                 ( 50,  50,  60)),
    ("crystal",   "a clear faceted crystal",                           (210, 230, 255)),
    ("gem",       "a glittering coloured gem",                         (255, 100, 150)),
    ("rock",      "a weathered grey rock",                             (130, 125, 120)),
    ("boulder",   "a massive boulder",                                 (120, 115, 110)),
    ("cliff",     "a sheer rocky cliff face",                          (130, 110,  90)),
    ("waterfall", "a cascading waterfall",                             (180, 210, 230)),
    ("river",     "a slow-flowing river",                              ( 90, 130, 170)),
    ("lake",      "a still mirror-like lake",                          (100, 140, 170)),
    ("ocean",     "deep open ocean",                                   ( 40,  80, 130)),
    ("cloud",     "fluffy white clouds",                               (230, 230, 230)),
    ("storm",     "a roiling storm front",                             ( 80,  85,  95)),
    ("lightning", "a forked lightning bolt",                           (240, 240, 255)),

    # Elements / FX
    ("fire",      "leaping orange flames",                             (240, 120,  40)),
    ("lava",      "molten flowing lava",                               (220,  80,  30)),
    ("ice",       "translucent blue ice",                              (200, 230, 255)),
    ("snow",      "drifting snow",                                     (250, 250, 255)),
    ("explosion", "a billowing explosion",                             (240, 160,  60)),
    ("beam",      "a focused energy beam",                             (180, 220, 255)),
    ("shield",    "a shimmering energy shield",                        (140, 200, 255)),
    ("aura",      "a glowing magical aura",                            (200, 160, 240)),
    ("trail",     "a streaking light trail",                           (220, 220, 240)),
    ("smoke",     "billowing dark smoke",                              ( 60,  60,  65)),
    ("fog",       "low-lying fog",                                     (200, 200, 210)),
    ("dust",      "drifting motes of dust",                            (180, 170, 140)),
    ("sparks",    "scattering bright sparks",                          (255, 230, 100)),
    ("bubbles",   "rising air bubbles",                                (220, 240, 255)),
    ("rain",      "steady falling rain",                               (180, 200, 220)),
    ("aurora",    "shimmering aurora curtains",                        (120, 240, 180)),
]


def seed() -> EntityVocabulary:
    v = EntityVocabulary()
    for n, s, c in SEED_ENTITIES:
        v.register(n, s, c)
    return v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_stats(v: EntityVocabulary):
    s = v.stats()
    print(f"=== EntityVocabulary ===")
    print(f"  total entities       : {s['entities']:>7d}  / capacity {s['capacity']:>10d}")
    print(f"  1-byte slots filled  : {s['1_byte_slots_used']:>7d}  / {EntityVocabulary.MAX_1BYTE}")
    print(f"  2-byte slots filled  : {s['2_byte_slots_used']:>7d}  / {EntityVocabulary.MAX_2BYTE_OFFSET}")
    print(f"  3-byte slots filled  : {s['3_byte_slots_used']:>7d}")
    print(f"  total scene_count    : {s['scene_count_total']}")
    print(f"  avg bytes per tag    : {s['avg_bytes_per_tag']}")


def _print_top(v: EntityVocabulary, n=20):
    print(f"\n--- top {n} by scene_count ---")
    print(f"  {'id':>5}  {'bytes':>5}  {'count':>6}  {'name':<14}  string")
    for i, e in enumerate(v.entities[:n]):
        print(f"  {i:>5}  {v.bytewidth_of(i):>5}  {e.scene_count:>6}  {e.name:<14}  {e.string}")


def main():
    ap = argparse.ArgumentParser(description="Lyra 2 Lite entity vocabulary tool.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("show", help="show vocabulary stats + top-N")
    sp.add_argument("--vocab", type=Path, help="vocab.json (default: built-in seed)")
    sp.add_argument("--top", type=int, default=20)

    sp = sub.add_parser("test", help="round-trip encode/decode test")
    sp.add_argument("--names", nargs="+", required=True)
    sp.add_argument("--vocab", type=Path)

    sp = sub.add_parser("save-seed", help="save the seed vocabulary to disk")
    sp.add_argument("--out", type=Path, default=Path("./entity_vocab.json"))

    sp = sub.add_parser("simulate", help="simulate frequency counts + resort")
    sp.add_argument("--scenes", type=int, default=200,
                    help="number of synthetic scenes to roll")
    sp.add_argument("--seed-rng", type=int, default=42)

    sp = sub.add_parser("prompt", help="build a prompt from entity names")
    sp.add_argument("--names", nargs="+", required=True)
    sp.add_argument("--vocab", type=Path)

    args = ap.parse_args()

    if args.cmd == "show":
        v = EntityVocabulary.load(args.vocab) if args.vocab else seed()
        _print_stats(v)
        _print_top(v, args.top)

    elif args.cmd == "test":
        v = EntityVocabulary.load(args.vocab) if args.vocab else seed()
        enc = v.encode_many(args.names)
        dec = v.decode_names(enc)
        widths = [v.bytewidth_of(n) for n in args.names]
        print(f"input ({len(args.names)} entities)  : {args.names}")
        print(f"per-entity byte widths             : {widths}")
        print(f"encoded ({len(enc)} bytes)         : {enc.hex(' ')}")
        print(f"decoded                            : {dec}")
        assert dec == args.names, "ROUND-TRIP FAILED"
        print("round-trip OK")

    elif args.cmd == "save-seed":
        seed().save(args.out)
        print(f"wrote {args.out}")

    elif args.cmd == "simulate":
        import random
        rng = random.Random(args.seed_rng)
        v = seed()
        # Zipf-ish weights so a few entities dominate, others rare.
        names = [e.name for e in v.entities]
        weights = [1.0 / (i + 1) ** 1.2 for i in range(len(names))]
        # Roll synthetic scene usage:
        for _ in range(args.scenes):
            n_entities = rng.randint(3, 12)
            picks = rng.choices(names, weights=weights, k=n_entities)
            v.increment_many(picks)
        print(f"simulated {args.scenes} scenes")
        _print_stats(v)
        print("\nBefore resort:")
        _print_top(v, 12)
        remap = v.resort()
        print(f"\nAfter resort (remap covers {len(remap)} entities)")
        _print_top(v, 12)
        # Sanity-check a sample stream round-trip survives the resort.
        sample = ["sky", "ground", "dolphin", "station"]
        # Pre-resort encoding (impossible to demo cleanly here -- show after-resort instead):
        enc = v.encode_many(sample)
        dec = v.decode_names(enc)
        print(f"\npost-resort round-trip {sample}")
        print(f"  encoded {len(enc)} bytes: {enc.hex(' ')}")
        print(f"  decoded: {dec}")

    elif args.cmd == "prompt":
        v = EntityVocabulary.load(args.vocab) if args.vocab else seed()
        prompt = v.prompt_for(args.names)
        print(prompt)


if __name__ == "__main__":
    main()
