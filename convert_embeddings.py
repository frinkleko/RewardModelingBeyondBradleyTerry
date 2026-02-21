"""
Convert embedding data from the embedding-based-llm-alignment repo format
(https://github.com/holarissun/embedding-based-llm-alignment) into the
split-based format expected by steps 5-8.

Input format (repo):
  EMBD file:   shape (K, N, D)  e.g. (10, 41876, 2048)
  REWARD file: shape (K, N)     e.g. (10, 41876)

Output format (local):
  data/EMBD-TRAIN-split_{k}.npy    shape (N, D)   for k in [0..K-1]
  data/rewards_split_{k}.json      JSONL with {"reward": float} per line

Usage:
  python convert_embeddings.py \
      --embd_file embd/RM-Embeddings/EMBD-TRAIN-..._gemma2b_...npy \
      --reward_file embd/RM-Embeddings/REWARD-TRAIN-..._gemma2b_...npy \
      --output_dir data

  python convert_embeddings.py --task harmless --gen_model gemma2b
"""
import os
import argparse
import glob
import numpy as np
import json


def find_embd_file(embd_dir, task, sft_obj, embed_model, gen_model):
    """Auto-detect the embedding file based on task/model names."""
    rm_name = "rmmistral7b" if "helpful" in task else "rayharmless"
    sft_suffix = "-gpt4" if sft_obj == "gpt4" else ""

    # Try the full original filename pattern first
    pattern_full = os.path.join(
        embd_dir,
        f"EMBD-TRAIN-embd_model_name_{embed_model}_dataset_orig_hh-rlhf-"
        f"{task}{sft_suffix}_gen_pref_model_name_{gen_model}_golden_rm_name_{rm_name}.npy"
    )
    if os.path.exists(pattern_full):
        return pattern_full

    # Try glob fallback
    candidates = glob.glob(os.path.join(embd_dir, f"EMBD-TRAIN*{task}*{gen_model}*.npy"))
    if candidates:
        return candidates[0]
    return None


def find_reward_file(embd_dir, task, sft_obj, embed_model, gen_model):
    """Auto-detect the reward file."""
    rm_name = "rmmistral7b" if "helpful" in task else "rayharmless"
    sft_suffix = "-gpt4" if sft_obj == "gpt4" else ""

    # Try full original filename
    pattern_full = os.path.join(
        embd_dir,
        f"REWARD-TRAIN-embd_model_name_{embed_model}_dataset_orig_hh-rlhf-"
        f"{task}{sft_suffix}_gen_pref_model_name_{gen_model}_golden_rm_name_{rm_name}.npy"
    )
    if os.path.exists(pattern_full):
        return pattern_full

    # Try short names (from curl downloads)
    short_name = f"REWARD-TRAIN-{task}{sft_suffix}-{gen_model}.npy"
    short_path = os.path.join(embd_dir, short_name)
    if os.path.exists(short_path):
        return short_path

    # Glob fallback
    candidates = glob.glob(os.path.join(embd_dir, f"REWARD-TRAIN*{task}*{gen_model}*.npy"))
    if candidates:
        return candidates[0]
    return None


def convert(embd_file, reward_file, output_dir, max_prompts=None):
    """Convert repo format to per-split local format."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading embeddings from: {embd_file}")
    embd_data = np.load(embd_file)  # shape (K, N, D)
    K, N, D = embd_data.shape
    print(f"  Embedding shape: ({K}, {N}, {D})")

    print(f"Loading rewards from: {reward_file}")
    reward_data = np.load(reward_file)  # shape (K, N)
    K_r, N_r = reward_data.shape
    print(f"  Reward shape: ({K_r}, {N_r})")

    assert K == K_r, f"K mismatch: embedding {K} vs reward {K_r}"
    assert N == N_r, f"N mismatch: embedding {N} vs reward {N_r}"

    if max_prompts and max_prompts < N:
        print(f"  Truncating to {max_prompts} prompts (from {N})")
        embd_data = embd_data[:, :max_prompts, :]
        reward_data = reward_data[:, :max_prompts]
        N = max_prompts

    for k in range(K):
        # Save embedding split
        emb_out = os.path.join(output_dir, f"EMBD-TRAIN-split_{k}.npy")
        np.save(emb_out, embd_data[k])  # shape (N, D)
        print(f"  Saved {emb_out} -> shape {embd_data[k].shape}")

        # Save reward split as JSONL
        rew_out = os.path.join(output_dir, f"rewards_split_{k}.json")
        with open(rew_out, "w") as f:
            for idx in range(N):
                json.dump({"reward": float(reward_data[k, idx])}, f)
                f.write("\n")
        print(f"  Saved {rew_out} -> {N} entries")

    print(f"\nConversion complete: {K} splits, {N} prompts, {D}-dim embeddings")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert embedding repo data to local format")
    parser.add_argument("--embd_file", type=str, default=None,
                        help="Path to EMBD .npy file (auto-detected if not provided)")
    parser.add_argument("--reward_file", type=str, default=None,
                        help="Path to REWARD .npy file (auto-detected if not provided)")
    parser.add_argument("--embd_dir", type=str, default="embd/RM-Embeddings",
                        help="Directory with downloaded embedding repo files")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for converted files")
    parser.add_argument("--task", type=str, default="harmless",
                        choices=["helpful", "harmless"],
                        help="Task name for auto-detection")
    parser.add_argument("--sft_obj", type=str, default="gpt4",
                        help="SFT objective (gpt4 or none)")
    parser.add_argument("--embed_model", type=str, default="gemma2b",
                        help="Embedding model name")
    parser.add_argument("--gen_model", type=str, default="gemma2b",
                        help="Response generation model name")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit number of prompts (for quick testing)")
    args = parser.parse_args()

    if args.embd_file is None:
        args.embd_file = find_embd_file(
            args.embd_dir, args.task, args.sft_obj, args.embed_model, args.gen_model
        )
        if args.embd_file is None:
            print(f"ERROR: Could not find EMBD file for task={args.task}, "
                  f"gen_model={args.gen_model} in {args.embd_dir}/")
            print("Available files:")
            for f in sorted(os.listdir(args.embd_dir)):
                if f.startswith("EMBD"):
                    print(f"  {f}")
            return

    if args.reward_file is None:
        args.reward_file = find_reward_file(
            args.embd_dir, args.task, args.sft_obj, args.embed_model, args.gen_model
        )
        if args.reward_file is None:
            print(f"ERROR: Could not find REWARD file for task={args.task}, "
                  f"gen_model={args.gen_model} in {args.embd_dir}/")
            print("Available files:")
            for f in sorted(os.listdir(args.embd_dir)):
                if f.startswith("REWARD"):
                    print(f"  {f}")
            return

    convert(args.embd_file, args.reward_file, args.output_dir, args.max_prompts)


if __name__ == "__main__":
    main()
