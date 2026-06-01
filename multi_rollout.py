"""Run N rollouts per variant with different seeds, saving to numbered subdirectories.

Before running: set DEFAULT_XML in env.py to the correct morphology for the variant.

Usage:
    python multi_rollout.py --variant variantBaseline --n 5 --noise 0.2
"""

import argparse
import os
import subprocess
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, help="Base variant name (e.g. variantBaseline)")
    p.add_argument("--n",       type=int, default=5, help="Number of rollouts")
    p.add_argument("--noise",   type=float, default=0.2)
    p.add_argument("--no-video", action="store_true", default=True)
    return p.parse_args()

def main():
    args = parse_args()
    ckpt = os.path.abspath(f"runs/{args.variant}/checkpoints/final")

    for seed in range(args.n):
        name = f"{args.variant}/r{seed}"
        cmd = [
            sys.executable, "rollout.py", name,
            "--checkpoint", ckpt,
            "--seed", str(seed),
            "--noise", str(args.noise),
            "--no-video",
        ]
        print(f"\n=== Rollout {seed+1}/{args.n}: {name} ===")
        subprocess.run(cmd, check=True)

    print(f"\nDone. Runs saved to: runs/{args.variant}/r0 ... runs/{args.variant}/r{args.n - 1}")

if __name__ == "__main__":
    main()
