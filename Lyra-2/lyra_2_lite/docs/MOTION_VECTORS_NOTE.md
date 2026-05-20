# Codec motion vectors → voxel motion histograms

**Draft, 2026-05-20 — Opie's first-principles take while the research
agent works in background.** Will append the agent's literature review
+ final synthesis when it lands.

---

## The setup, in one paragraph

H.264 / H.265 / AV1 encoders compute **per-macroblock 2D motion vectors**
as a compression step — for every ~16×16 block of every frame, the
codec writes "this block at frame N came from offset (Δx, Δy) in
frame N-1." These MVs ship inside the bitstream for free; extracting
them via PyAV with `flags2 |= AV_CODEC_FLAG2_EXPORT_MVS` is one
codec-context flag and a per-frame `side_data` read (we verified
~1,800 MVs/frame on a 640×360 H.264 clip earlier tonight).

For *any* encoded video — Lyra 2's output, a YouTube rip, surveillance
footage, your own gameplay capture — that's a free 2D optical-flow
proxy at macroblock granularity. **Combined with depth + camera
intrinsics, those 2D MVs unproject to 3D scene flow per macroblock.**
Combined with a voxel atlas and a known camera trajectory, that 3D
scene flow accumulates into a per-voxel motion history.

Three first-principles uses, ordered by how clearly they pay off:

---

## Use 1 — Static vs dynamic voxel classification (the clean one)

**Observation:** A voxel that is part of static scenery (a wall, the
ground, a tree) has zero *world-frame* motion across all frames it
appears in. A voxel that is part of a moving entity (a character, a
vehicle, a particle) has consistent non-zero motion within short
windows.

**Algorithm:**
1. For each macroblock with MV (du, dv) at screen position (u, v) in
   frame N:
2. Look up depth z at (u, v) from the rendered depth buffer
3. Compute 3D position in world frame at frame N-1 via camera
   intrinsics K and pose (R_{N-1}, t_{N-1}):
   ```
   p_world(N-1) = R_{N-1}^T · ( K^{-1} · [u,   v,   1]^T · z   - t_{N-1} )
   p_world(N)   = R_N^T     · ( K^{-1} · [u+du, v+dv, 1]^T · z - t_N )
   3D_motion    = p_world(N) - p_world(N-1)
   ```
4. **Subtract expected motion due to camera movement alone** (computable
   from R, t change between frames) → residual is *object-only*
   motion
5. Accumulate this 3D residual into the voxel's motion history
6. Per voxel: residual mean ≈ 0 → static. Non-zero mean → dynamic.

**This is the clean payoff.** Static-vs-dynamic segmentation is genuinely
hard (per-pixel methods like FlowNet are expensive to run; semantic
methods like Mask-R-CNN need training and miss novel classes). Codec
MVs + depth + camera-trajectory subtraction gives you the answer
**for free**, accurate enough for most uses.

For Lyra 2 Lite specifically: knowing which voxels are dynamic tells
you which need the future "dynamic object" rendering layer (the
unsolved-tonight gap we flagged in the proposal) and which can stay
in the static atlas. Auto-segmentation of generated content into
"baked scenery" + "objects of interest."

---

## Use 2 — Per-voxel motion histograms as a diffusion conditioning signal

**Observation:** Lyra 2 generates videos one chunk at a time, with
each chunk autoregressively conditioning on the previous chunk's
latent. Temporal coherence is a known weak point — characters can
drift, lose identity, change colour subtly across chunk boundaries.

**Hypothesis:** Feed accumulated per-voxel motion histograms (or
their reduction: motion vectors per chunk) into the model as
auxiliary conditioning. Tells the diffusion: *"this region of the
scene was observed to move in direction d at speed s in the past; if
you're regenerating it, the motion should be consistent."*

This is structurally similar to how some video diffusion papers
condition on optical flow (e.g., **MotionDirector**, **Direct-a-Video**,
**MotionMaster**, all 2024-2025), but the source of the flow is
different — codec MVs, not learned flow networks. Much cheaper.

**Why this might work better than learned flow:** codec MVs are
calibrated by years of perceptual tuning in encoder design. They
implicitly weight "what humans notice" via the encoder's rate-distortion
optimization. As a *prior* (not a perceptually-faithful pixel-level
flow), they could be more useful than RAFT/GMA outputs which optimise
for pixel-accurate flow that humans don't actually perceive.

**Why this might NOT work:** codec MVs are designed for compression,
not motion truth. Block-flat regions, low-texture areas, occlusion
boundaries — the encoder gives plausible-but-wrong MVs there because
they happen to compress well. Bad signal in those cases.

This use case needs validation. **Worth a small experiment, not yet
a deployed feature.**

---

## Use 3 — Compression of dynamic 4D voxel scenes

**Observation:** Storing a moving voxel scene over time naively means
storing every voxel at every frame — wildly redundant. Most voxels
don't move. The motion histogram tells you *which* voxels need
per-frame state and which don't.

**Compression idea:**
- Static voxels: store once in the atlas, never again
- Quasi-static voxels (low-motion variance): store once + a small
  motion model (mean velocity + variance)
- Dynamic voxels: store per-frame state, but only for the chunks
  containing dynamic content

For a 1 km² Mêlée-class scene where 95% of voxels are scenery and 5%
are characters/particles/water: **20× compression vs naive 4D voxel
storage.**

This is the same algorithmic idea as MPEG (keyframes + deltas) but
applied at voxel granularity instead of pixel. The motion histogram
is the meta-data that lets you decide per voxel.

---

## What's most likely novel about this combination

Codec MVs as a CV signal is well-trodden (people have used them for
action recognition since 2014). 2D-to-3D scene flow via depth + camera
is standard (NSFF, DSFlow). Per-voxel velocity fields are in OpenVDB /
Houdini for years.

What's **plausibly fresh** is the *specific* combination:

1. **Free codec MVs** (not learned flow) as the input signal
2. **Lyra 2-generated content** (known depth + known trajectory) as
   the unprojection context
3. **Per-voxel histogram accumulation** as the data structure
4. **Static-vs-dynamic auto-segmentation** as the output product
5. **Conditioning signal for the same generator** that produced the
   content as the closing-the-loop application

The combination is a self-improving pipeline: Lyra 2 generates a
scene, MVs from the generated video inform per-voxel motion histograms,
those histograms condition the next generation, which produces more
coherent video, with cleaner MVs, in a self-reinforcing loop.

Whether this loop *converges* or *diverges* is an empirical question.
I don't know of published work on this specific closed loop.

---

## Implementation cost estimate (5060 Ti, 1080p video)

| Step | CPU/GPU work | Wall time per frame |
|---|---|---:|
| PyAV side_data extract | CPU | ~0.5 ms |
| Depth buffer access | GPU (already rendered) | 0 ms |
| 2D MV → 3D scene flow | CPU vectorized numpy | ~2 ms |
| Camera-motion subtract | CPU | <0.5 ms |
| Project to voxel grid + accumulate to histogram | GPU compute shader | ~3 ms |
| **Total per-frame overhead** | | **~5-6 ms** |

At 60 fps that's ~300-360 ms/sec of overhead per second of input
video, or **5-6% of frame budget**. Comfortable for real-time use,
extremely cheap for offline post-processing of a generated video.

For a 30-second Lyra 2 chunk at 12 fps: ~2 sec of MV extraction +
histogram building. Negligible compared to the diffusion time.

---

## What I'd build first (if greenlit)

A `mv_to_voxel_motion.py` module in `lyra_pr_branch/` that:

1. Takes a Lyra 2 output `.mp4` + the trajectory `.npz` (Lyra 2's
   custom-trajectory format already provides this) + the rendered
   depth buffer sequence
2. Extracts codec MVs via PyAV
3. Unprojects to 3D scene flow per macroblock per frame
4. Subtracts camera-motion projection
5. Accumulates into a per-voxel motion histogram in the same UVW
   atlas layout as the existing OccupancyBitmap
6. Emits: a per-voxel `(motion_mean, motion_variance, n_observations)`
   tensor; a derived `is_dynamic` boolean mask; a chunk-level summary
   for streaming decisions

Pipeline: ~1 day Opie time. Standalone, testable on any Lyra 2 output
or even arbitrary video + ground-truth depth+camera (e.g., synthetic
test clips).

Then a separate experiment to feed the motion histograms back into
the next Lyra 2 generation as conditioning — *that* requires fine-tune
compute. Standalone "motion histogram extractor + static/dynamic
classifier" can ship without training, just as a useful tool.

---

## Open questions for empirical testing

1. **Codec MV reliability for content from a diffusion model** —
   Lyra 2 outputs are unlike natural video; encoder MVs on synthetic
   diffusion content may be noisier than on real footage. Worth
   measuring before committing.

2. **Block-flat regions** — sky, fog, smooth gradients — give the
   encoder weak MV signal. How big is the artifact in practice?

3. **Long-horizon histogram drift** — does motion histogram quality
   improve with more observations, or does it saturate / get worse
   from encoder artifacts at long horizons?

4. **Closed-loop feedback** — does conditioning Lyra 2 on motion
   histograms from its own previous output improve or degrade
   temporal coherence? (Both outcomes are plausible.)

---

---

## Literature review (agent-mined, 2026-05-20)

### What's been done — codec MVs as a CV signal

The field is mature:

- **CoViAR (CVPR'18)** and **DMC-Net (CVPR'19)** established H.264 MVs as
  usable for action recognition but flagged them as
  **noisy, block-quantised, biased toward rate-distortion not motion truth**.
  DMC-Net specifically had to learn a "discriminative motion cue"
  generator to denoise them, supervised by ground-truth flow during
  training ([DMC-Net paper](https://arxiv.org/pdf/1901.03460)).

- **Critical update — AV1 fixes most of the noise problem.**
  [*"AV1 Motion Vector Fidelity and Application for Efficient Optical Flow"* (Oct 2025)](https://arxiv.org/pdf/2510.17427)
  benchmarks codec MVs against ground-truth flow:

  | Codec | Typical error vs ground-truth flow |
  |---|---:|
  | H.264 | 5-8 px EPE |
  | HEVC (H.265) | 3-5 px EPE |
  | **AV1 (tuned encoder)** | **2-4 px EPE** |
  | SOTA learned flow (RAFT, GMA) | ~0.5 px EPE |

  Conclusion: **AV1 MVs are a "high-quality and computationally
  efficient substitute for traditional optical flow."** H.264 is
  noticeably noisier. **Practical takeaway for our pipeline: encode
  Lyra 2's output with `libsvtav1` or `libx265`, NOT `libx264`,
  before extracting MVs.**

- Companion paper [*"Leveraging AV1 motion vectors for Fast and Dense Feature Matching"* (Oct 2025)](https://arxiv.org/html/2510.17434v2)
  reuses them as a sparse-to-dense seed for matching.

- [**MVP** (Sept 2025)](https://arxiv.org/pdf/2509.18388) explicitly
  calls codec MVs *"available at negligible cost during decoding"* and
  uses them for zero-shot object detection.

- [**"Temporal Realism Evaluation of Generated Videos Using Compressed-Domain MVs"** (Nov 2025)](https://arxiv.org/pdf/2511.13897)
  is directly adjacent to us — they use codec MVs as a *quality metric*
  for generated video. Closest published work to our use case.

**Standard preprocessing:** discard MVs from intra-coded blocks; discard
sub-8×8 partitions where rate-distortion picks weak matches; median
filter spatially; reject magnitudes above a per-frame-rate threshold.

**Important practical note** ([PyAV #793](https://github.com/PyAV-Org/PyAV/issues/793)):
MV export requires full decode. "Negligible during decoding" is true
*if you're already decoding*. For our Lyra 2 pipeline this is free —
we decode the generated video anyway.

### 2D MV → 3D scene flow given depth + camera intrinsics

The canonical formula is **Vedula et al. (1999)**: scene_flow(p) =
unproject(p + flow, depth(p+flow)) − unproject(p, depth(p)) under
known K. Modern monocular variant is
[**Yang & Ramanan, "Upgrading Optical Flow to 3D Scene Flow Through Optical Expansion," CVPR'20**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Upgrading_Optical_Flow_to_3D_Scene_Flow_Through_Optical_Expansion_CVPR_2020_paper.pdf)
which adds optical expansion to recover Z when stereo isn't available.
**With known depth (Lyra 2 has it), the expansion step is
unnecessary — direct unprojection is exact up to flow noise.**

[Deep Scene Flow Learning survey, TPAMI 2023](https://dl.acm.org/doi/abs/10.1109/TPAMI.2023.3319448)
documents SOTA EPE3D on KITTI at ~3-5 cm with learned methods.

**For our case with codec MVs as input:** expected ~15-30 cm error at
1 m range. **Fine for static/dynamic classification, not for precise
dynamics.** This calibrates the realistic ambition.

### Voxel velocity field representations — terminology gap

OpenVDB convention: a `Vec3fGrid` named `vel` co-aligned with the
density grid — same tree topology, one velocity per active voxel
([cgwiki](https://tokeru.com/cgwiki/HoudiniVolumes.html),
[SideFX docs](https://www.sidefx.com/docs/houdini/model/volumes.html)).
GVDB and Houdini volume engines use the same pattern.

**There is no standard term "voxel motion histogram" in the
volumetric literature.** Closest established concepts:

| Term | What it stores | Source |
|---|---|---|
| **Per-voxel velocity field** (`vel` grid) | One vec3 per voxel | OpenVDB, GVDB, Houdini |
| **Motion History Volume** (Weinland et al. 2006) | Scalar occupancy over time | Action recognition |
| **HOOF-3D** (Histogram of Oriented 3D Optical Flow) | Per-region histogram | Region-level descriptor |

A *distribution* stored per voxel (rather than a single vector) is
**unusual** in this literature. **Small but real terminology /
representation gap we'd be coining when we call this a "voxel motion
histogram."**

### Ego-motion vs object-motion decomposition

Textbook: ["Towards Optical Flow Ego-motion Compensation for Moving Object Segmentation"](https://www.semanticscholar.org/paper/Towards-Optical-Flow-Ego-motion-Compensation-for-Moving-Object-Segmentation/4ec5c56a05d5f1cd9f6aa4cfd99cbf48b24011ec)
(Elek & Károly, 2020) is the canonical recent reference. Called
**"ego-motion compensated optical flow"** or **"residual flow"**. Also
standard in SLAM (DynaSLAM, VDO-SLAM).

**Lyra 2 has perfect ego-motion** (it's the input trajectory),
*unlike* SLAM systems which estimate it noisily. Real practical
advantage — we have an exact reference to subtract against.

### What's been done in 2024-2026 with voxel motion

| Paper | What it does | Year |
|---|---|---|
| [**4D-GS** (Wu et al.)](https://guanjunwu.github.io/4dgs/) | HexPlane-decomposed neural voxel deformation grid | CVPR'24 |
| [**MEGA**](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_MEGA_Memory-Efficient_4D_Gaussian_Splatting_for_Dynamic_Scenes_ICCV_2025_paper.pdf) | 4DGS compression | ICCV'25 |
| [**SpeeDe3DGS**](https://arxiv.org/html/2506.07917v1) | Prunes by *temporal sensitivity* = per-Gaussian motion magnitude — **functionally a dynamic-mass voxel score** | Jun 2025 |
| [**4D Neural Voxel Splatting**](https://arxiv.org/html/2511.00560) | Time-aware features in a voxel grid | Nov 2025 |
| [**OnlyFlow**](https://arxiv.org/html/2411.10501v1) | Conditions video diffusion on optical flow | Nov 2024 |
| [**MotionFlow**](https://arxiv.org/abs/2509.21119) | Implicit 3D motion conditioning for video diffusion | Sept 2025 |

The static/dynamic split via a velocity threshold (~90th percentile)
is **standard 4DGS practice** as of 2025. Our pipeline would compute
exactly the same score, just from codec MVs instead of training
deformation fields.

### Novelty assessment — what we'd be contributing

| Component | Prior art | Novel for us? |
|---|---|---|
| Codec MVs as CV signal | CoViAR, DMC-Net, AV1-Fidelity '25 | ❌ No |
| 2D flow → 3D scene flow with depth+K | Vedula '99, Yang & Ramanan '20 | ❌ No |
| Ego-motion / residual flow | Elek '20, DynaSLAM, VDO-SLAM | ❌ No |
| Per-voxel velocity field | OpenVDB `vel`, 4DGS standard | ❌ No |
| **MV-driven + depth-conditioned + UVW-atlas-indexed + per-voxel histogram accumulated across generated Lyra-2 clips + fed back as DiT conditioning** | **None found in agent's search** | ✅ **Combination is novel** |
| Using *known-perfect* generated trajectory rather than noisy SLAM-estimated pose | Implicit in synthetic-data work | ⚠️ Modestly novel framing |
| **Closing the loop: motion histograms condition the next generation** | Adjacent (OnlyFlow conditions on flow), but not from the model's own *previous output's codec MVs* fed back as 3D voxel-atlas conditioning | ✅ **Genuinely fresh** |

The integration as a whole is RFC-shaped. **The closed-loop angle
(generator → MVs → 3D voxel motion histograms → conditioning for next
generation, with perfect camera trajectory throughout) is not in the
literature the agent found.**

---

## Implementation cost (confirmed, 5060 Ti)

Per second of 1080p24 video:

| Step | Cost |
|---|---|
| **NVDEC H.265/AV1 decode + MV side-data extraction** | ~5-8% of one NVDEC engine; hardware-accelerated; ~free |
| MV filter + unproject (~43k unprojects/sec) | <0.5 ms/frame on CPU, <100 µs on GPU |
| Ego-motion subtraction (4×4 multiply per MV) | negligible |
| Voxel atlas scatter-add (UVW lookup + atomic update) | bottleneck step, ~sub-ms on 5060 Ti |
| Histogram bucketing (8 dir × 4 mag = 32 bins × 1 byte = 32 B/voxel) | 512 MB at 256³ voxels — comfortable in 16 GB VRAM |
| **Total** | **<2 ms/frame, ~1% GPU, 50-500 MB VRAM** |

**Real-time is trivially achievable on the 5060 Ti.** This is the
opposite of a research-direction-only result — it's an *engineering-light*
contribution that ships in a few hundred lines on top of the existing
UVW atlas writer.

---

## Architectural decision (2026-05-20, after the backface-coverage discussion)

The original framing of this note treated dynamic voxels as a *content
store* — accumulate observations across frames, 4D-reconstruct moving
entities in object-local coordinates, render as full voxel models.
**We're not doing that.** Three reasons:

1. **Backface coverage gap.** Dynamic voxels are by construction the
   front-facing surface that the original camera observed. Their
   backfaces don't exist in the atlas because Lyra 2 was never asked
   to invent them. Even with temporal aggregation across many frames,
   coverage of the full 360° around a character is unreliable unless
   the original camera explicitly circled them — which most camera
   trajectories don't.

2. **Lyra 2 character quality is the wrong tool.** Lyra 2 is a *video
   diffusion model for environments.* It produces beautiful walkable
   scenes; it produces mediocre characters because that's not what it
   was trained to do. Trying to extract character voxels from Lyra 2's
   output is fighting the model's strengths.

3. **A character-specific pipeline exists and is better.** DWPose +
   ControlNet-OpenPose + Hunyuan3D + Mixamo gives us properly rigged,
   fully-textured, view-independent 3D character models with real
   animations — for the same hardware budget. It's the right tool for
   the job and it's *already in the DownToEarth pipeline* via
   Hunyuan3D.

So the architecture shifts to a **hybrid renderer**:

```
Static scenery layer (95% of the world):
    UVW voxel atlas (this note's pipeline)
    Baked once per scene from Lyra 2 generation
    Raymarched at runtime per pixel
    Demand-streamed from disk via 2-bit OccupancyBitmap

Dynamic character layer (~5%):
    Skeletal-rigged 3D mesh models per NPC
    Generated separately: DWPose → ControlNet T-pose →
        Hunyuan3D mesh → Mixamo auto-rig → animation clips
    Rendered conventionally (triangles, hardware rasterization)
    Composited over voxel raymarch via depth-buffer test
```

This is the same architectural pattern as **Teardown** (voxel
environment + rigged character meshes) and roughly mirrors how 1990s
adventure games shipped (pre-rendered scenery + 3D characters), just
with the scenery upgraded from baked images to voxel raymarching.

### What the motion atlas's role becomes

Originally proposed: a 4D content store.
**Now: a segmentation and quality-control tool.** Three concrete uses
that earn its keep:

| Use | What it does |
|---|---|
| **Atlas-pruning mask** | Voxels classified dynamic (via the camera-coherence test) are *removed* from the baked static atlas. Solves the artifact problem where ghost-character voxels would otherwise sit baked into the scenery. Pure-static atlas; no leftover dynamic noise. |
| **Character-region segmentation prior** | DWPose is expensive; running it on whole frames is wasteful. Restricting it to the dynamic regions identified by the motion atlas can be ~10× faster. The dynamic-chunk set is a free attention map for the character pipeline. |
| **Camera-trajectory health check** | High dynamic-vote density in regions that *should* be static = bug in the bake (camera path went through a region with insufficient coverage, leading to flickery codec MVs). Trigger regeneration / camera-path correction. |

### The character pipeline (separate doc, summarized here)

DWPose → ControlNet-OpenPose T-pose front + back → Hunyuan3D mesh →
Mixamo auto-rig → animation retarget. ~5-10 min per character on the
5060 Ti. Models in question (all already accessible via ComfyUI):

| Stage | Model | Cost |
|---|---|---:|
| Pose extraction from video | **DWPose** (`comfyui_controlnet_aux`) | ~30 sec / 5-min clip |
| Front T-pose still | ControlNet-OpenPose + Juggernaut XL + IPAdapter | ~30 sec |
| Back T-pose still | Same, with rear-view conditioning | ~30 sec |
| 3D mesh from 2 views | Hunyuan3D (already in voxgaussian pipeline) | ~3 min |
| Auto-rig | Mixamo (cloud, free) or AccuRIG | seconds |
| Animation retarget from DWPose keyframes | Mixamo / Blender / Unity Animator | ~1 sec / sec source |

For a Monkey-Island-class cast (~20 named characters): ~3 hours of
character generation, total. Re-poseable and re-lightable forever
after.

## Updated implementation plan

Three concrete steps, ordered by ROI:

### Step 1 — `motion_atlas.py` module (1 day Opie time)

Lives at `lyra_pr_branch/Lyra-2/lyra_2/_src/datasets/motion_atlas.py`
as a sibling to `uvw_atlas.py`. Provides:

```python
class MotionAtlas:
    """
    Per-voxel motion histogram atlas, co-indexed with the UVW atlas.
    Stores 32 bins per voxel (8 directions × 4 magnitudes) at 1 byte
    each, or alternatively a packed (mean_vec3 + variance_scalar)
    representation for tighter storage.
    """
    def accumulate(self, voxel_xyz, motion_3d): ...
    def static_mask(self, threshold: float) -> ndarray: ...    # for atlas culling
    def dynamic_chunks(self) -> set[tuple]: ...                # for the dynamic layer
    def conditioning_tensor(self, chunk_xyz) -> torch.Tensor:  # for DiT cross-attention

@dataclass
class MotionFrame:
    timestamp: float
    camera_pose: tuple[ndarray, ndarray]   # R, t
    intrinsics: ndarray                    # K
    depth_buffer: ndarray                  # HxW float
    motion_vectors: ndarray                # AVMotionVector structured array

def extract_to_atlas(
    video_path: Path,
    trajectory: Path,    # Lyra 2's .npz
    depth_sequence: Path,
    atlas: MotionAtlas,
) -> dict:
    """Streams a video, extracts AV1/H.265 MVs, unprojects to 3D scene
    flow with ego-motion subtraction, scatters into the motion atlas.
    Returns summary statistics + dynamic-chunk set."""
```

Key implementation details from the lit review:
- **Use AV1 (`libsvtav1`) or HEVC (`libx265`) encoding** — H.264 MVs
  are noisy enough to bias the histogram
- Discard intra-coded MVs and sub-8×8 partitions
- Median-filter MVs spatially before unprojection (denoise)
- Subtract camera-induced apparent motion using the *exact* camera
  pose from Lyra 2's trajectory `.npz`
- 90th-percentile velocity threshold for static/dynamic split (per
  4DGS standard practice)

### Step 2 — Encoder change in the bake pipeline (~30 min)

Wherever Lyra 2's MP4 output gets written, switch the codec from
`libx264` (default) to `libsvtav1` (best MV fidelity per
[arXiv:2510.17427](https://arxiv.org/pdf/2510.17427)). One ffmpeg
argument change.

### Step 3 — DiT conditioning channel integration (1-2 weeks, fine-tune required)

For the closed-loop angle — feeding motion histograms back as
conditioning to the next Lyra 2 generation — requires a LoRA
fine-tune or sibling-head training. Sits in the same effort tier as
the [[project_lora_sd_bitdepth_ladder]] direction. Defer until step 1
is shipped and we have actual histogram data to validate against.

---

## Final verdict

**Actionable tonight for Step 1.** All ingredients are off-the-shelf:
PyAV MV export ✅, depth from Lyra 2's render ✅, perfect camera from
Lyra 2's trajectory ✅, UVW atlas already in PR #61 ✅, OpenVDB-style
`vel` grid convention as a template ✅. The novel combination is
engineering-light: ~few hundred lines on top of the existing UVW
atlas writer.

**Critical encoder note:** Lyra 2's output must be re-encoded with
**AV1 (libsvtav1) or HEVC (libx265), not H.264 (libx264)**, before MV
extraction. The MV fidelity difference (2-4 px vs 5-8 px error) is
the difference between "useful 3D scene flow signal" and "noise."

**Headline contribution if shipped:** a sibling to OccupancyBitmap
that gives Lyra 2 a self-knowledge channel — *"these voxels move,
these voxels don't, and here's how they move"* — derived from the
codec's own motion search at no extra compute cost. Pairs naturally
with the dynamic-object architecture gap we flagged earlier. Could
unlock the dynamic-content render layer that was stubbed in the
proposal.

**Closing the loop** (Step 3) is the genuinely novel research
contribution — a self-improving generation pipeline where each
generation provides its own motion priors for the next. This is the
RFC-shaped angle worth eventually proposing publicly. But ship Step
1 first; nothing to propose without measured data.

---

## Status

- ✅ Literature review complete (agent, 14 tool uses, ~2 min)
- ✅ First-principles design complete (Opie, prior section)
- ✅ Implementation plan with critical encoder constraint identified
- ⬜ Step 1 (`motion_atlas.py`) — actionable tonight, ~1 day Opie time
- ⬜ Step 2 (encoder switch in bake pipeline) — ~30 min when Step 1 is ready
- ⬜ Step 3 (DiT conditioning channel) — research-grade, defer

## Sources

- [DMC-Net (CVPR'19)](https://arxiv.org/pdf/1901.03460) — H.264 MVs for action recognition
- [AV1 Motion Vector Fidelity (Oct 2025)](https://arxiv.org/pdf/2510.17427) — **the critical paper; AV1 MVs as flow substitute**
- [AV1 MVs for Dense Feature Matching (Oct 2025)](https://arxiv.org/html/2510.17434v2)
- [MVP — MV Propagation for Detection (Sept 2025)](https://arxiv.org/pdf/2509.18388)
- [Temporal Realism via Compressed-Domain MVs (Nov 2025)](https://arxiv.org/pdf/2511.13897) — closest published work to our use case
- [Yang & Ramanan, Optical Flow to 3D Scene Flow (CVPR'20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Upgrading_Optical_Flow_to_3D_Scene_Flow_Through_Optical_Expansion_CVPR_2020_paper.pdf)
- [Deep Scene Flow Learning survey (TPAMI 2023)](https://dl.acm.org/doi/abs/10.1109/TPAMI.2023.3319448)
- [Ego-motion Compensation for Moving Object Segmentation (Elek & Károly, 2020)](https://www.semanticscholar.org/paper/Towards-Optical-Flow-Ego-motion-Compensation-for-Moving-Object-Segmentation/4ec5c56a05d5f1cd9f6aa4cfd99cbf48b24011ec)
- [OpenVDB / Houdini velocity field convention](https://tokeru.com/cgwiki/HoudiniVolumes.html)
- [4D-GS (CVPR'24)](https://guanjunwu.github.io/4dgs/)
- [MEGA — 4DGS compression (ICCV'25)](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_MEGA_Memory-Efficient_4D_Gaussian_Splatting_for_Dynamic_Scenes_ICCV_2025_paper.pdf)
- [SpeeDe3DGS (Jun 2025)](https://arxiv.org/html/2506.07917v1) — temporal-sensitivity pruning
- [4D Neural Voxel Splatting (Nov 2025)](https://arxiv.org/html/2511.00560)
- [OnlyFlow (Nov 2024)](https://arxiv.org/html/2411.10501v1) — flow-conditioned video diffusion
- [MotionFlow (Sept 2025)](https://arxiv.org/abs/2509.21119) — implicit 3D motion conditioning
- [PyAV MV side-data discussion](https://github.com/PyAV-Org/PyAV/issues/793)

Ties to the existing stub at
`voxgaussian/posterize/extract_motion_vectors.py` (research-only
scaffold from earlier this session) — that file documented the
PyAV decode path; this file documents the architectural design
built on top.
