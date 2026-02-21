import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import glob
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from networks import MLP

parser = argparse.ArgumentParser()
parser.add_argument("--embed_model_name", type=str, default="gemma2b")
parser.add_argument("--task", type=str, default="helpful")
parser.add_argument("--sft_obj", type=str, default="gpt4")
parser.add_argument("--output_dir", type=str, default="distill_out")
parser.add_argument(
    "--data_dir",
    type=str,
    default="data",
    help="Directory containing EMBD-TRAIN-split_*.npy and rewards_split_*.json",
)
parser.add_argument("--ensemble_number", type=int, default=10)
parser.add_argument("--rm_objective", type=str, choices=["clf", "bt"], default="bt")
parser.add_argument("--consider_first_n", type=int, default=2)
parser.add_argument("--n_sample", type=int, default=100)
parser.add_argument("--n_pairs", type=int, default=1)
parser.add_argument("--training_epochs", type=int, default=30)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument(
    "--replacement",
    type=str,
    default="replacement_false",
    choices=["replacement_true", "replacement_false"],
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
args_replacement = True if args.replacement == "replacement_true" else False

def load_calibration_data(data_dir):
    """Load embeddings and rewards from local .npy/.json files."""
    emb_files = sorted(glob.glob(os.path.join(data_dir, "EMBD-TRAIN-split_*.npy")))
    rew_files = sorted(glob.glob(os.path.join(data_dir, "rewards_split_*.json")))
    if not emb_files or not rew_files:
        return None, None

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
    
    # We'll use pairwise differences for BT calibration
    return np.array(all_embeddings), np.array(all_rewards)

print("=" * 60)
print(f"Calibrating: task={args.task}, objective={args.rm_objective}")
print("=" * 60)

embeddings, rewards = load_calibration_data(args.data_dir)
if embeddings is None:
    print(f"ERROR: No data found in {args.data_dir}/")
    exit(1)

n_splits, n_prompts, emb_dim = embeddings.shape

# Use the last model in the ensemble for calibration demo
model_i = args.ensemble_number - 1
ckpt_name = (
    f"{args.output_dir}/XPrompt_mlp_{args.rm_objective}_{model_i}_seed{args.seed}_firstn_"
    f"{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_"
    f"replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.ckpt"
)

if not os.path.exists(ckpt_name):
    print(f"ERROR: Model checkpoint not found: {ckpt_name}")
    exit(1)

model = MLP(emb_dim)
model.load_state_dict(torch.load(ckpt_name, map_location="cpu", weights_only=True))
model.eval()

# Generate pairwise samples for calibration
all_scores_diff = []
all_labels = []

print("Generating predictions for calibration...")
for p_idx in range(n_prompts):
    for s1 in range(n_splits):
        for s2 in range(s1 + 1, n_splits):
            emb1 = torch.tensor(embeddings[s1, p_idx], dtype=torch.float32).unsqueeze(0)
            emb2 = torch.tensor(embeddings[s2, p_idx], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                score1 = model(emb1).item()
                score2 = model(emb2).item()
            
            all_scores_diff.append(score1 - score2)
            all_labels.append(1 if rewards[s1][p_idx] > rewards[s2][p_idx] else 0)

            # Also add the reverse pair
            all_scores_diff.append(score2 - score1)
            all_labels.append(1 if rewards[s2][p_idx] > rewards[s1][p_idx] else 0)

X = np.array(all_scores_diff)
y = np.array(all_labels)

# Use Bradley-Terry sigmoid as base prob
base_probs = 1.0 / (1.0 + np.exp(-np.clip(X, -20, 20)))

# Fit Isotonic Regression
print("Fitting Isotonic Regression...")
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(base_probs, y)

# Save the calibrator
calibrator_fn = (
    f"{args.output_dir}/isotonic_calibrator_{args.rm_objective}_{model_i}_seed{args.seed}.joblib"
)
joblib.dump(iso_reg, calibrator_fn)
print(f"Calibrator saved to: {calibrator_fn}")

# Evaluate calibration
calibrated_probs = iso_reg.predict(base_probs)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_idx = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        if np.any(bin_idx):
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            ece += np.abs(bin_acc - bin_conf) * np.sum(bin_idx) / len(y_true)
    return ece

ece_before = expected_calibration_error(y, base_probs)
ece_after = expected_calibration_error(y, calibrated_probs)

print(f"ECE Before Calibration: {ece_before:.4f}")
print(f"ECE After Calibration: {ece_after:.4f}")

# Save calibration results
cal_results = {
    "ece_before": float(ece_before),
    "ece_after": float(ece_after),
}
cal_res_fn = (
    f"{args.output_dir}/calibration_results_{args.rm_objective}_{model_i}_seed{args.seed}.json"
)
with open(cal_res_fn, "w") as f:
    json.dump(cal_results, f, indent=2)
