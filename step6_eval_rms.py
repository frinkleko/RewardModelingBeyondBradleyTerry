import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import argparse
import json
import glob
from networks import MLP, forward_siamese, train_model, save_model
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--embed_model_name", type=str, default="gemma2b")
parser.add_argument("--task", type=str, default="helpful")
parser.add_argument("--sft_obj", type=str, default="gpt4")
parser.add_argument("--output_dir", type=str, default="distill_out")
parser.add_argument("--data_dir", type=str, default="data",
                    help="Directory containing EMBD-TRAIN-split_*.npy and rewards_split_*.json")
parser.add_argument("--server_alias", type=str, default="lq")
parser.add_argument("--gen_pref_model_name", type=str, default="gemma7b")
parser.add_argument("--ensemble_number", type=int, default=10)
parser.add_argument("--rm_objective", type=str, choices=["clf", "bt"], default="bt")
parser.add_argument("--consider_first_n", type=int, default=2)
parser.add_argument("--n_sample", type=int, default=100)
parser.add_argument("--n_pairs", type=int, default=1)
parser.add_argument("--training_epochs", type=int, default=30)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--normal_or_xprompt", type=str, default="xprompt", choices=["normal", "xprompt"])
parser.add_argument("--replacement", type=str, default="replacement_false",
                    choices=["replacement_true", "replacement_false"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_len", type=int, default=-1)
parser.add_argument("--annotation_quality", type=float, default=10)
args = parser.parse_args()
args.sft_obj_name = "none" if args.sft_obj == "" or args.sft_obj == "none" else "gpt4"
args.sft_obj_suffix = "" if args.sft_obj == "none" else "-gpt4"
args_replacement = True if args.replacement == "replacement_true" else False
rm_name = "rmmistral7b" if args.task == "helpful" else "rayharmless"

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_local_eval_data(data_dir):
    """Load embeddings and rewards from local .npy/.json files for evaluation."""
    emb_files = sorted(glob.glob(os.path.join(data_dir, "EMBD-TRAIN-split_*.npy")))
    rew_files = sorted(glob.glob(os.path.join(data_dir, "rewards_split_*.json")))
    if not emb_files or not rew_files:
        return None, None
    
    # Use all available splits for evaluation
    all_embeddings = []
    all_rewards = []
    for emb_f, rew_f in zip(emb_files, rew_files):
        emb = np.load(emb_f)
        all_embeddings.append(emb)
        rewards = []
        with open(rew_f) as f:
            for line in f:
                data = json.loads(line)
                rewards.append(data["reward"])
        all_rewards.append(rewards)
    
    # Stack: shape [n_splits, n_prompts, emb_dim]
    # For evaluation, we treat each split as a different response per prompt
    n_splits = len(all_embeddings)
    n_prompts = all_embeddings[0].shape[0]
    emb_dim = all_embeddings[0].shape[1]
    
    # Transpose to [n_prompts, n_splits, emb_dim] for per-prompt evaluation
    stacked_emb = np.stack(all_embeddings, axis=0)  # [n_splits, n_prompts, emb_dim]
    stacked_rew = np.array(all_rewards)  # [n_splits, n_prompts]
    
    # reward_list[prompt_idx] = [reward_for_split_0, reward_for_split_1, ...]
    reward_list = []
    for p in range(n_prompts):
        reward_list.append([stacked_rew[s, p] for s in range(n_splits)])
    
    return stacked_emb, reward_list


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("=" * 60)
print(f"Evaluating: task={args.task}, objective={args.rm_objective}")
print("=" * 60)

# Load evaluation data
stacked_emb, reward_list = load_local_eval_data(args.data_dir)
if stacked_emb is None:
    print(f"ERROR: No evaluation data found in {args.data_dir}/")
    exit(1)

n_splits = stacked_emb.shape[0]
n_prompts = stacked_emb.shape[1]
emb_dim = stacked_emb.shape[2]
print(f"Loaded {n_splits} splits, {n_prompts} prompts, embedding dim {emb_dim}")

# Load trained models
model_predictions = []
for model_i in range(args.ensemble_number - 1, args.ensemble_number):
    ckpt_name = (
        f"{args.output_dir}/XPrompt_mlp_{args.rm_objective}_{model_i}_seed{args.seed}_firstn_"
        f"{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_"
        f"replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.ckpt"
    )
    if not os.path.exists(ckpt_name):
        print(f"Warning: Model checkpoint not found: {ckpt_name}")
        continue

    model = MLP(emb_dim)
    model.load_state_dict(torch.load(ckpt_name, map_location="cpu", weights_only=True))
    model.eval()

    model_i_all_preds = []
    for prompt_idx in range(n_prompts):
        preds_per_split = []
        for split_idx in range(n_splits):
            emb_tensor = torch.tensor(
                stacked_emb[split_idx, prompt_idx], dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                pred = model(emb_tensor).item()
            preds_per_split.append(pred)
        model_i_all_preds.append(preds_per_split)
    model_predictions.append(model_i_all_preds)

if not model_predictions:
    print("ERROR: No models loaded. Run step5 first.")
    exit(1)

model_predictions = np.array(model_predictions)  # [n_models, n_prompts, n_splits]
model_predictions_mean = model_predictions.mean(0)  # [n_prompts, n_splits]

# Calculate Spearman correlations
correlations = []
for p_idx in range(n_prompts):
    if n_splits < 3:
        # With only 2 splits, Spearman is degenerate; use sign agreement instead
        pred_order = np.argsort(model_predictions_mean[p_idx])
        true_order = np.argsort(reward_list[p_idx])
        correlations.append(1.0 if np.array_equal(pred_order, true_order) else 0.0)
    else:
        corr = spearmanr(model_predictions_mean[p_idx], reward_list[p_idx]).correlation
        correlations.append(corr)

correlations = np.array(correlations)
print(f"\nMean ranking agreement: {np.nanmean(correlations):.4f}")

# Pairwise accuracy: for each prompt, check if model correctly ranks split_0 vs split_1
if n_splits >= 2:
    correct = 0
    total = 0
    for p_idx in range(n_prompts):
        for s1 in range(n_splits):
            for s2 in range(s1 + 1, n_splits):
                pred_diff = model_predictions_mean[p_idx][s1] - model_predictions_mean[p_idx][s2]
                true_diff = reward_list[p_idx][s1] - reward_list[p_idx][s2]
                if (pred_diff > 0) == (true_diff > 0):
                    correct += 1
                total += 1
    pairwise_acc = correct / total if total > 0 else 0.0
    print(f"Pairwise accuracy: {pairwise_acc:.4f} ({correct}/{total})")

# Save results
results = {
    "task": args.task,
    "rm_objective": args.rm_objective,
    "n_prompts": n_prompts,
    "n_splits": n_splits,
    "mean_ranking_agreement": float(np.nanmean(correlations)),
    "pairwise_accuracy": float(pairwise_acc) if n_splits >= 2 else None,
    "correlations": correlations.tolist(),
}

result_fn = (
    f"{args.output_dir}/eval_results_{args.rm_objective}_seed{args.seed}_firstn_"
    f"{args.consider_first_n}_n_{args.n_sample}.json"
)
with open(result_fn, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {result_fn}")
