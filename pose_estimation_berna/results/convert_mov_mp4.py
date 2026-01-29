#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Convert a .mov file to .mp4 using ffmpeg")
    p.add_argument("input", help="Path to input .mov")
    p.add_argument("-o", "--output", help="Path to output .mp4 (default: same name, .mp4)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH. Install it first (e.g., `sudo apt-get install ffmpeg`).")

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path = Path(args.output) if args.output else in_path.with_suffix(".mp4")

    cmd = [
        "ffmpeg",
        "-y" if args.overwrite else "-n",
        "-i", str(in_path),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()