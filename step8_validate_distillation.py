import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import glob
import joblib
from scipy.stats import spearmanr
from networks import MLP
from tqdm import tqdm


def predict_batch(model, embeddings, device, batch_size=512):
    """Run model inference in batches, return numpy array."""
    model.eval()
    preds = []
    for i in range(0, len(embeddings), batch_size):
        batch = torch.tensor(embeddings[i : i + batch_size], dtype=torch.float32).to(
            device
        )
        with torch.no_grad():
            p = model(batch).cpu().numpy().flatten()
        preds.extend(p)
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_model_name", type=str, default="gemma2b")
    parser.add_argument("--task", type=str, default="helpful")
    parser.add_argument("--sft_obj", type=str, default="gpt4")
    parser.add_argument("--gen_pref_model_name", type=str, default="gemma2b")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--student_ckpt", type=str, required=True)
    parser.add_argument(
        "--isotonic_model",
        type=str,
        required=True,
        help="Teacher isotonic map (isotonic_teacher_map.joblib or isotonic_map.joblib)",
    )
    parser.add_argument(
        "--student_isotonic_model",
        type=str,
        default="",
        help="Student isotonic map (isotonic_student_map.joblib). "
        "If empty, tries output_dir/isotonic_student_map.joblib",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing EMBD-TRAIN-split_*.npy and rewards_split_*.json",
    )
    parser.add_argument("--output_dir", type=str, default="distill_out")
    parser.add_argument("--server_alias", type=str, default="lq")
    parser.add_argument(
        "--normalize_rewards",
        action="store_true",
        help="Min-max normalize golden rewards to [0,1] (must match step7 setting)",
    )
    parser.add_argument(
        "--binarize_rewards",
        action="store_true",
        help="Convert golden rewards to binary 0/1 (must match step7 setting)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load evaluation data (structured by prompt/split for BoN)
    # ------------------------------------------------------------------
    print("Loading evaluation data...")
    emb_files = sorted(glob.glob(os.path.join(args.data_dir, "EMBD-TRAIN-split_*.npy")))
    rew_files = sorted(glob.glob(os.path.join(args.data_dir, "rewards_split_*.json")))

    if not emb_files or not rew_files:
        print(f"ERROR: No data found in {args.data_dir}/")
        return

    split_embeddings = []
    split_rewards = []
    for emb_f, rew_f in zip(emb_files, rew_files):
        emb = np.load(emb_f)
        with open(rew_f) as f:
            rewards = [json.loads(l)["reward"] for l in f]
        n = min(len(rewards), emb.shape[0])
        split_embeddings.append(emb[:n])
        split_rewards.append(np.array(rewards[:n]))

    # Shape: [n_splits, n_prompts, emb_dim]
    stacked_emb = np.stack(split_embeddings, axis=0)
    stacked_rew = np.stack(split_rewards, axis=0)
    n_splits, n_prompts, emb_dim = stacked_emb.shape
    print(f"Loaded {n_splits} splits, {n_prompts} prompts, dim = {emb_dim}")

    # Flatten for global metrics
    all_embeddings = stacked_emb.reshape(-1, emb_dim)
    all_golden = stacked_rew.reshape(-1)

    # Optional: binarize golden rewards
    if args.binarize_rewards:
        cfg_path = os.path.join(args.output_dir, "distillation_config.json")
        threshold = None
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            if "binarize_params" in cfg:
                threshold = cfg["binarize_params"]["threshold"]
        if threshold is None:
            threshold = float(np.median(all_golden))
        n_pos = int((all_golden > threshold).sum())
        all_golden = (all_golden > threshold).astype(np.float64)
        stacked_rew = (stacked_rew > threshold).astype(np.float64)
        print(
            f"Binarized golden rewards at threshold {threshold:.4f}: "
            f"{n_pos} positive ({100*n_pos/len(all_golden):.1f}%)"
        )

    # Optional: normalize golden rewards to [0,1]
    if args.normalize_rewards:
        cfg_path = os.path.join(args.output_dir, "distillation_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            if "norm_params" in cfg:
                r_min, r_max = cfg["norm_params"]["reward_min"], cfg["norm_params"]["reward_max"]
            else:
                r_min, r_max = float(all_golden.min()), float(all_golden.max())
        else:
            r_min, r_max = float(all_golden.min()), float(all_golden.max())
        
        all_golden = (all_golden - r_min) / (r_max - r_min + 1e-8)
        stacked_rew = (stacked_rew - r_min) / (r_max - r_min + 1e-8)
        print(f"Normalized golden rewards to [0,1] (range [{r_min:.4f}, {r_max:.4f}])")

    # ------------------------------------------------------------------
    # 2. Load models
    # ------------------------------------------------------------------
    print(f"Loading teacher from {args.teacher_ckpt}")
    teacher = MLP(emb_dim).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device, weights_only=True))

    print(f"Loading student from {args.student_ckpt}")
    student = MLP(emb_dim).to(device)
    student.load_state_dict(torch.load(args.student_ckpt, map_location=device, weights_only=True))

    print(f"Loading teacher isotonic model from {args.isotonic_model}")
    ir_teacher = joblib.load(args.isotonic_model)

    student_iso_path = args.student_isotonic_model or os.path.join(args.output_dir, "isotonic_student_map.joblib")
    ir_student = joblib.load(student_iso_path) if os.path.exists(student_iso_path) else None

    # ------------------------------------------------------------------
    # 3. Best-of-N (BoN) Evaluation
    # ------------------------------------------------------------------
    print("\nCalculating Best-of-N (BoN) Improvement...")
    
    # We use all prompts for BoN to get a stable estimate
    # Teacher scores: [n_splits, n_prompts]
    t_scores = predict_batch(teacher, all_embeddings, device).reshape(n_splits, n_prompts)
    s_scores = predict_batch(student, all_embeddings, device).reshape(n_splits, n_prompts)
    ti_scores = ir_teacher.transform(t_scores.flatten()).reshape(n_splits, n_prompts)
    si_scores = ir_student.transform(s_scores.flatten()).reshape(n_splits, n_prompts) if ir_student else None

    # Base Model performance (Average of random responses)
    base_avg_reward = np.mean(stacked_rew)

    def calculate_bon_improvement(scores, rewards):
        # For each prompt, pick the split with the highest score
        best_indices = np.argmax(scores, axis=0) # [n_prompts]
        best_rewards = rewards[best_indices, np.arange(n_prompts)]
        avg_best_reward = np.mean(best_rewards)
        improvement = avg_best_reward - base_avg_reward
        return avg_best_reward, improvement

    bon_results = {}
    variants = [("Teacher", t_scores), ("Teacher+Iso", ti_scores), ("Student", s_scores)]
    if si_scores is not None: variants.append(("Student+Iso", si_scores))

    print(f"\n{'Model':<20} {'Avg BoN Reward':>15} {'Improvement':>15}")
    print("-" * 55)
    print(f"{'Base Model':<20} {base_avg_reward:>15.4f} {0.0:>15.4f}")
    
    for name, scores in variants:
        avg_r, imp = calculate_bon_improvement(scores, stacked_rew)
        bon_results[name] = {"avg_reward": float(avg_r), "improvement": float(imp)}
        print(f"{name:<20} {avg_r:>15.4f} {imp:>15.4f}")

    # ------------------------------------------------------------------
    # 4. Global Metrics (Spearman, MSE, Pair Accuracy)
    # ------------------------------------------------------------------
    print("\nCalculating Global Metrics (Order Consistency)...")
    
    # Use held-out 20% for these metrics to avoid over-optimism
    n_val = max(1, int(len(all_embeddings) * 0.2))
    val_emb = all_embeddings[-n_val:]
    val_golden = all_golden[-n_val:]
    
    t_val = t_scores.flatten()[-n_val:]
    ti_val = ti_scores.flatten()[-n_val:]
    s_val = s_scores.flatten()[-n_val:]
    si_val = si_scores.flatten()[-n_val:] if si_scores is not None else None

    sp = lambda a, b: float(spearmanr(a, b).correlation)
    
    rows = [("Teacher", t_val), ("Teacher+Iso", ti_val), ("Student", s_val)]
    if si_val is not None: rows.append(("Student+Iso", si_val))

    print(f"\n{'Model':<20} {'Spearman':>10} {'Pair Acc':>10}")
    print("-" * 45)
    
    np.random.seed(42)
    n_pairs = min(5000, len(val_emb) * (len(val_emb) - 1) // 2)
    idx_i = np.random.randint(0, len(val_emb), n_pairs)
    idx_j = np.random.randint(0, len(val_emb), n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    golden_order = val_golden[idx_i] > val_golden[idx_j]

    final_metrics = {}
    for name, preds in rows:
        rho = sp(preds, val_golden)
        acc = float(np.mean((preds[idx_i] > preds[idx_j]) == golden_order))
        final_metrics[name] = {"spearman": rho, "pair_acc": acc, "bon": bon_results[name]}
        print(f"{name:<20} {rho:>10.4f} {acc:>10.4f}")

    # ------------------------------------------------------------------
    # 5. Scale check (sample predictions)
    # ------------------------------------------------------------------
    n_show = min(5, len(val_emb))
    headers = ["GoldenTruth", "Teacher", "T+Isotonic", "Student"]
    if si_preds is not None:
        headers.append("S+Isotonic")

    print(f"\nScale Check (first {n_show} validation samples):")
    print("".join(f"{h:>14}" for h in headers))
    for i in range(n_show):
        vals = [val_golden[i], t_preds[i], ti_preds[i], s_preds[i]]
        if si_preds is not None:
            vals.append(si_preds[i])
        print("".join(f"{v:>14.4f}" for v in vals))

    # ------------------------------------------------------------------
    # 6. Save validation results
    # ------------------------------------------------------------------
    out = {
        "base_model_avg_reward": float(base_avg_reward),
        "metrics": final_metrics
    }
    result_path = os.path.join(args.output_dir, "validation_results.json")
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nValidation results saved to {result_path}")


if __name__ == "__main__":
    main()
