#!/usr/bin/env python3
"""Generate synthetic demo data for CPU-only pipeline testing.

Creates fake embeddings (2048-dim) and reward scores so that steps 5-8
can be tested without downloading real embedding data.

Usage:
    python3 generate_demo_data.py                       # default: data/
    python3 generate_demo_data.py --output_dir my_data  # custom dir
"""

import os
import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo embeddings + rewards"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Directory to write demo data"
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=10,
        help="Number of splits (analogous to 10 generated responses per prompt)",
    )
    parser.add_argument(
        "--n_prompts", type=int, default=200, help="Number of prompts per split"
    )
    parser.add_argument("--emb_dim", type=int, default=2048, help="Embedding dimension")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(
        f"Generating demo data: {args.n_splits} splits × {args.n_prompts} prompts × {args.emb_dim}D"
    )

    for k in range(args.n_splits):
        # Each split has a shared "prompt signal" + random "response variation"
        prompt_signal = (
            np.random.randn(args.n_prompts, args.emb_dim).astype(np.float32) * 0.5
        )
        response_var = (
            np.random.randn(args.n_prompts, args.emb_dim).astype(np.float32) * 0.3
        )
        emb = prompt_signal + response_var

        # Rewards: correlated with first few embedding dimensions (simulates real structure)
        reward = (
            emb[:, 0] * 2.0
            + emb[:, 1] * 1.5
            - emb[:, 2] * 1.0
            + np.random.randn(args.n_prompts) * 0.5
        )

        emb_path = os.path.join(args.output_dir, f"EMBD-TRAIN-split_{k}.npy")
        rew_path = os.path.join(args.output_dir, f"rewards_split_{k}.json")

        np.save(emb_path, emb)
        with open(rew_path, "w") as f:
            for r in reward:
                json.dump({"reward": float(r)}, f)
                f.write("\n")

    print(f"Demo data written to {args.output_dir}/")
    print(
        f"  {args.n_splits} × EMBD-TRAIN-split_*.npy  ({args.n_prompts}, {args.emb_dim})"
    )
    print(f"  {args.n_splits} × rewards_split_*.json     (JSONL)")


if __name__ == "__main__":
    main()
