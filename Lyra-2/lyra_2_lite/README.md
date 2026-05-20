# Lyra 2 Lite

This subtree is the comprehensive "Lyra 2 Lite" project — the architectural
extensions, tooling, and demos built on top of NVIDIA's Lyra 2.0 to make
photo-to-3D-world generation work on consumer GPUs.

The upstream contribution to NVIDIA's Lyra 2 codebase lives in the parent
directory at `lyra_2/` and is filed as
[**nv-tlabs/lyra#61**](https://github.com/nv-tlabs/lyra/pull/61). This
`lyra_2_lite/` subtree is the broader project — research demos, tooling,
postprocessing, and documentation that sits *on top of* that upstream
contribution.

## What's here

```
lyra_2_lite/
├── entity_vocab.py              # Variable-byte (1/2/3 byte) priority encoding for
│                                # entity tags. Self-tuning vocabulary; scene_count
│                                # drives the byte-width assignment. Used as
│                                # structured-prompt conditioning + per-voxel
│                                # semantic ID + downstream segment renderer ID.
├── structured_prompt.py         # StructuredPromptBuilder + lyra2_caption_hook
│                                # drop-in replacement for raw caption input.
│                                # Auto-extracts entity tokens, expands the
│                                # caption with canonical entity descriptions,
│                                # emits entity-byte stream + color hints.
├── posterize/                   # Bit-depth ladder dataset pipeline (per
│                                # LYRA2_PROPOSAL §6.6.2). Scrape, build ladder
│                                # rungs 1-BPP→8-BPP per RGB channel, train
│                                # TinyUNet deposterizer, sample multi-step.
│   ├── posterize.py             #   LUT-based ladder at 168 fps (1080p)
│   ├── scrape_dataset.py        #   16-worker picsum scrape, 22.8 img/s
│   ├── build_rgb_rungs.py       #   per-channel RGB rungs from existing dataset
│   ├── deposterizer.py          #   TinyUNet trainer (1ch luma / 3ch RGB modes)
│   ├── noise_schedule.py        #   Bounded-uniform noise (±N per rung)
│   ├── sample.py                #   Multi-step inference, rung 1→8 chain
│   └── eval_all.py              #   Held-out batch eval
├── voxel_renderer/              # WebGL2 single-file demo of the runtime
│                                # raymarch pipeline. DDA voxel traversal +
│                                # per-chunk frustum culling. 60+ fps on 5060 Ti.
├── postprocess/
│   └── rife_upsample.py         # RIFE 4.26 wrapper for 12 fps → 60 fps after
│                                # Lyra 2 generation. Native fp16 on Blackwell.
├── scripts/
│   └── setup_lyra2_wsl.sh       # End-to-end WSL2 Ubuntu setup for running
│                                # Lyra 2 inference on a Windows + 5060 Ti host.
└── docs/
    ├── PROPOSAL.md              # The public RFC (also filed as PR #61)
    ├── DUMMIES.md               # Layperson explanation (metaphors throughout)
    ├── MOTION_VECTORS_NOTE.md   # Codec-MV → voxel-motion-histogram design
    │                              # + literature review + implementation plan
    └── SESSION_TODO.md          # Active development tracker
```

## Relationship to the upstream PR

The patches in PR #61 (against NVIDIA's `lyra_2/` tree) are kept
deliberately **surgical** — they touch only the canonical-coord
encoding path, the OccupancyBitmap, and the consumer-GPU low-VRAM
loader. NVIDIA's Lyra 2 codebase doesn't grow a posterize pipeline or
a WebGL voxel renderer; those are research / tooling that sit
alongside.

This `lyra_2_lite/` subtree is **never** part of PR #61. It's the home
of the broader project that consumes Lyra 2 + the PR-61 changes to
produce a consumer-GPU pipeline.

## License

This subtree is original work by MiLO and contributed under
**Apache 2.0** to match the upstream Lyra 2 fork.

The files that originated in the DownToEarth project
([github.com/MiLO83/DownToEarth](https://github.com/MiLO83/DownToEarth),
MIT-licensed) have been relicensed to Apache 2.0 for inclusion here
(both licenses are permissive; MIT is downgradable to Apache without
issue, and MiLO holds copyright on the original work either way).

## Public artifacts that ship from this work

- **Live demo + canonical proposal hosting**:
  [downtoearth-9lq.pages.dev](https://downtoearth-9lq.pages.dev)
- **Architectural RFC**: [PROPOSAL.md](docs/PROPOSAL.md) (also
  published at the URL above)
- **Layperson explainer**: [DUMMIES.md](docs/DUMMIES.md)

## Quick links inside the fork

- **PR #61 commits** (the surgical upstream contribution): see the
  `feat/uvw-bijection-canonical-encoding` branch
- **This subtree's commit history**: this `lyra-2-lite` branch
- **NVIDIA's original Lyra 2 code**: `../lyra_2/` (untouched)

## Building

The posterize + voxel_renderer + entity_vocab modules have no
inter-dependencies and can be used independently. See each
subdirectory's README for build / run details.

The end-to-end pipeline (Lyra 2 inference → RIFE upsample → voxel atlas
bake → runtime raymarch) requires the full Lyra 2 setup; see
[`scripts/setup_lyra2_wsl.sh`](scripts/setup_lyra2_wsl.sh) for the
WSL2-on-Windows path or NVIDIA's [INSTALL.md](../INSTALL.md) for
upstream Ubuntu.
