"""
RIFE post-processor for Lyra 2 Lite output.

The Lyra 2 / Lyra 2 Lite pipeline generates 12-fps native chunks (the
"animation on twos" rate). This script bumps any output video to a
target framerate via RIFE 4.26 interpolation -- typically 2x (12->24,
cinematic), 4x (12->48, smooth), or 5x (12->60, VR-grade).

Wraps Practical-RIFE's inference_video.py with cleaner defaults and
graceful failure on the audio-transfer step (which is broken on
moviepy 2.x and irrelevant for Lyra 2 output which has no audio anyway).

Usage:
    python rife_upsample.py --input scene.mp4 --multi 4
    python rife_upsample.py --input scene.mp4 --target-fps 60   # picks multi automatically
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
RIFE_DIR = HERE / "Practical-RIFE"


def probe_fps(video_path: Path) -> float:
    """Get input video's framerate via ffprobe."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of",
         "default=noprint_wrappers=1:nokey=1", str(video_path)],
        text=True,
    ).strip()
    num, den = (int(x) for x in out.split("/"))
    return num / max(den, 1)


def pick_multi(in_fps: float, target_fps: float) -> int:
    """Pick the smallest integer multiplier that meets/exceeds target."""
    if in_fps <= 0:
        raise ValueError("input fps reported as zero")
    raw = target_fps / in_fps
    return max(2, int(round(raw)))


def run_rife(input_video: Path, output_video: Path, multi: int,
             fp16: bool = True, uhd: bool = False) -> None:
    if not RIFE_DIR.exists():
        raise SystemExit(f"Practical-RIFE not found at {RIFE_DIR}. "
                         "Clone with: git clone https://github.com/hzwer/Practical-RIFE.git")
    if not (RIFE_DIR / "train_log" / "flownet.pkl").exists():
        raise SystemExit(f"RIFE weights not found at {RIFE_DIR}/train_log/flownet.pkl. "
                         "Download RIFE_4.26.zip and extract.")

    # inference_video.py writes <name>_<multi>X_<outfps>fps.mp4 next to the input
    # and crashes on its audio-transfer step (moviepy.editor moved in v2.x).
    # We accept that and rename the result ourselves.
    in_path = input_video.resolve()
    work_dir = in_path.parent

    cmd = [
        sys.executable, str(RIFE_DIR / "inference_video.py"),
        f"--video={in_path}",
        f"--multi={multi}",
    ]
    if fp16:
        cmd.append("--fp16")
    if uhd:
        cmd.append("--UHD")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(RIFE_DIR), capture_output=True, text=True)
    elapsed = time.time() - t0

    # Filter the inference output; suppress only the known moviepy.editor crash.
    if proc.returncode != 0 and "moviepy.editor" not in (proc.stdout + proc.stderr):
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise SystemExit(f"RIFE failed with code {proc.returncode}")

    # Find the produced output file. inference_video.py names it
    # <basename>_<multi>X_<outfps>fps.mp4 . Same directory as input.
    stem = in_path.stem
    candidates = list(work_dir.glob(f"{stem}_{multi}X_*fps.mp4"))
    if not candidates:
        raise SystemExit(f"RIFE produced no output. stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    produced = max(candidates, key=lambda p: p.stat().st_mtime)
    if produced.resolve() != output_video.resolve():
        shutil.move(str(produced), str(output_video))
    print(f"[rife] {in_path.name} ({multi}x) -> {output_video.name}  ({elapsed:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description="RIFE 4.26 framerate upsampler for Lyra 2 Lite output.")
    ap.add_argument("--input", type=Path, required=True, help="input video (any codec ffmpeg can read)")
    ap.add_argument("--output", type=Path, help="output video path (default: <input>_rife.mp4)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--multi", type=int, help="interpolation multiplier (2/4/5/8). Default 4.")
    grp.add_argument("--target-fps", type=float, help="target framerate; multi inferred from input fps")
    ap.add_argument("--no-fp16", action="store_true", help="disable fp16 (use full fp32; slower)")
    ap.add_argument("--uhd", action="store_true", help="enable UHD mode for >=4K inputs")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"missing: {args.input}")
    in_fps = probe_fps(args.input)
    if args.target_fps is not None:
        multi = pick_multi(in_fps, args.target_fps)
        print(f"[rife] input {in_fps:.1f} fps  target {args.target_fps:.1f} fps  -> {multi}x")
    else:
        multi = args.multi or 4
        print(f"[rife] input {in_fps:.1f} fps  multi {multi}x  -> {in_fps*multi:.1f} fps")
    if multi < 2:
        raise SystemExit("multi must be >= 2")

    out_path = args.output or args.input.with_name(
        args.input.stem + f"_rife{multi}x" + args.input.suffix
    )
    run_rife(args.input, out_path, multi=multi, fp16=not args.no_fp16, uhd=args.uhd)


if __name__ == "__main__":
    main()
