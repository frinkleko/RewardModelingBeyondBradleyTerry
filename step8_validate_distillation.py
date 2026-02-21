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
        batch = torch.tensor(
            embeddings[i:i+batch_size], dtype=torch.float32
        ).to(device)
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
    parser.add_argument("--isotonic_model", type=str, required=True,
                        help="Teacher isotonic map (isotonic_teacher_map.joblib or isotonic_map.joblib)")
    parser.add_argument("--student_isotonic_model", type=str, default="",
                        help="Student isotonic map (isotonic_student_map.joblib). "
                             "If empty, tries output_dir/isotonic_student_map.joblib")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing EMBD-TRAIN-split_*.npy and rewards_split_*.json")
    parser.add_argument("--output_dir", type=str, default="distill_out")
    parser.add_argument("--server_alias", type=str, default="lq")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load evaluation data
    # ------------------------------------------------------------------
    print("Loading evaluation data...")
    emb_files = sorted(glob.glob(os.path.join(args.data_dir, "EMBD-TRAIN-split_*.npy")))
    rew_files = sorted(glob.glob(os.path.join(args.data_dir, "rewards_split_*.json")))

    if not emb_files or not rew_files:
        print(f"ERROR: No data found in {args.data_dir}/")
        return

    all_embeddings, all_golden = [], []
    for emb_f, rew_f in zip(emb_files, rew_files):
        emb = np.load(emb_f)
        with open(rew_f) as f:
            rewards = [json.loads(l)["reward"] for l in f]
        n = min(len(rewards), emb.shape[0])
        all_embeddings.append(emb[:n])
        all_golden.extend(rewards[:n])

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_golden = np.array(all_golden)
    input_dim = all_embeddings.shape[-1]
    print(f"Loaded {len(all_embeddings)} samples, dim = {input_dim}")

    # Use last 20% as held-out validation
    n_val = max(1, int(len(all_embeddings) * 0.2))
    val_emb = all_embeddings[-n_val:]
    val_golden = all_golden[-n_val:]
    print(f"Using last {n_val} samples for validation")

    # ------------------------------------------------------------------
    # 2. Load models
    # ------------------------------------------------------------------
    print(f"Loading teacher from {args.teacher_ckpt}")
    teacher = MLP(input_dim).to(device)
    teacher.load_state_dict(
        torch.load(args.teacher_ckpt, map_location=device, weights_only=True)
    )

    print(f"Loading student from {args.student_ckpt}")
    student = MLP(input_dim).to(device)
    student.load_state_dict(
        torch.load(args.student_ckpt, map_location=device, weights_only=True)
    )

    print(f"Loading teacher isotonic model from {args.isotonic_model}")
    ir_teacher = joblib.load(args.isotonic_model)

    # Try to load student isotonic model
    student_iso_path = args.student_isotonic_model
    if not student_iso_path:
        student_iso_path = os.path.join(args.output_dir, "isotonic_student_map.joblib")
    ir_student = None
    if os.path.exists(student_iso_path):
        print(f"Loading student isotonic model from {student_iso_path}")
        ir_student = joblib.load(student_iso_path)
    else:
        print(f"No student isotonic model found at {student_iso_path} — skipping Student+Isotonic")

    # ------------------------------------------------------------------
    # 3. Evaluate all variants
    # ------------------------------------------------------------------
    print("\nEvaluating models on validation set...")

    t_preds = predict_batch(teacher, val_emb, device)
    s_preds = predict_batch(student, val_emb, device)
    ti_preds = ir_teacher.transform(t_preds)
    si_preds = ir_student.transform(s_preds) if ir_student is not None else None

    sp = lambda a, b: float(spearmanr(a, b).correlation)
    mse_fn = lambda a, b: float(np.mean((a - b) ** 2))

    results = {}
    rows = [
        ("Teacher ORM",        t_preds),
        ("Teacher + Isotonic", ti_preds),
        ("Student",            s_preds),
    ]
    if si_preds is not None:
        rows.append(("Student + Isotonic", si_preds))

    print(f"\n{'='*65}")
    print(f"Results ({len(val_emb)} validation samples)")
    print(f"{'='*65}")
    print(f"{'Model':<25} {'Spearman':>10} {'MSE':>10}")
    print(f"{'-'*50}")
    for name, preds in rows:
        s = sp(preds, val_golden)
        m = mse_fn(preds, val_golden)
        results[name] = {"spearman": s, "mse": m}
        print(f"{name:<25} {s:>10.4f} {m:>10.4f}")

    # ------------------------------------------------------------------
    # 4. Pairwise accuracy (does the model rank pairs correctly?)
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"Pairwise Accuracy (random 10000 pairs)")
    print(f"{'='*65}")
    np.random.seed(42)
    n_pairs = min(10000, len(val_emb) * (len(val_emb) - 1) // 2)
    i_idx = np.random.randint(0, len(val_emb), n_pairs)
    j_idx = np.random.randint(0, len(val_emb), n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    golden_order = val_golden[i_idx] > val_golden[j_idx]

    print(f"{'Model':<25} {'Pair Accuracy':>15}")
    print(f"{'-'*42}")
    for name, preds in rows:
        pred_order = preds[i_idx] > preds[j_idx]
        acc = float(np.mean(pred_order == golden_order))
        results[name]["pair_accuracy"] = acc
        print(f"{name:<25} {acc:>15.4f}")

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
        "n_validation_samples": len(val_emb),
        **{k: v for k, v in results.items()},
    }
    result_path = os.path.join(args.output_dir, "validation_results.json")
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nValidation results saved to {result_path}")


if __name__ == "__main__":
    main()
