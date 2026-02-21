import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import json
import glob
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split, KFold
from networks import MLP, train_model, save_model
from scipy.stats import spearmanr
from tqdm import tqdm


# ======================================================================
# Helpers
# ======================================================================

def load_local_data(data_dir, max_samples=None):
    """Load EMBD-TRAIN-split_*.npy + rewards_split_*.json from data_dir."""
    emb_files = sorted(glob.glob(os.path.join(data_dir, "EMBD-TRAIN-split_*.npy")))
    rew_files = sorted(glob.glob(os.path.join(data_dir, "rewards_split_*.json")))
    if not emb_files or not rew_files:
        return None, None
    all_emb, all_rew = [], []
    for ef, rf in zip(emb_files, rew_files):
        emb = np.load(ef)
        with open(rf) as f:
            rewards = [json.loads(l)["reward"] for l in f]
        n = min(len(rewards), emb.shape[0])
        all_emb.append(emb[:n])
        all_rew.extend(rewards[:n])
        if max_samples and len(all_rew) >= max_samples:
            break
    embeds = np.concatenate(all_emb, axis=0)
    rewards = np.array(all_rew)
    if max_samples and len(rewards) > max_samples:
        embeds, rewards = embeds[:max_samples], rewards[:max_samples]
    return embeds, rewards


def predict_batch(model, embeddings, device, batch_size=512):
    """Run model inference in batches, return numpy array."""
    model.eval()
    preds = []
    for i in range(0, len(embeddings), batch_size):
        batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            p = model(batch).cpu().numpy().flatten()
        preds.extend(p)
    return np.array(preds)


def cv_isotonic_fit(teacher_preds, golden_scores, n_folds=5, seed=42):
    """
    Cross-validated isotonic regression: fit on (K-1) folds, predict on
    held-out fold. Returns out-of-fold calibrated predictions AND a final
    isotonic model refit on all data.

    This avoids overfitting that makes isotonic degenerate on unseen data.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_calibrated = np.zeros_like(golden_scores)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(teacher_preds)):
        ir_fold = IsotonicRegression(out_of_bounds="clip")
        ir_fold.fit(teacher_preds[train_idx], golden_scores[train_idx])
        oof_calibrated[val_idx] = ir_fold.transform(teacher_preds[val_idx])

    # Final model refit on all data (used at inference time)
    ir_full = IsotonicRegression(out_of_bounds="clip")
    ir_full.fit(teacher_preds, golden_scores)

    return oof_calibrated, ir_full


def train_student(model, X_train, y_train, X_val, y_val,
                  epochs, lr, batch_size, device, label="Student"):
    """Train a student MLP with MSE loss. Returns the trained model."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_vl = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    patience = 5

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_tr.size(0))
        epoch_loss, n_b = 0.0, 0
        for i in range(0, X_tr.size(0), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_vl), y_vl).item()

        print(f"  [{label}] Epoch {epoch+1}/{epochs}, "
              f"Loss: {epoch_loss/max(n_b,1):.4f}, Val MSE: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{label}] Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Isotonic Distillation with cross-validated calibration "
                    "and Student + Student-Isotonic variants."
    )
    parser.add_argument("--embed_model_name", type=str, default="gemma2b")
    parser.add_argument("--task", type=str, default="helpful")
    parser.add_argument("--sft_obj", type=str, default="gpt4")
    parser.add_argument("--gen_pref_model_name", type=str, default="gemma2b")
    parser.add_argument("--rm_objective", type=str, choices=["clf", "bt"], default="bt")
    parser.add_argument(
        "--teacher_ckpt", type=str, required=True,
        help="Path to the trained ORM checkpoint from step5",
    )
    parser.add_argument("--output_dir", type=str, default="distill_out")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing EMBD-TRAIN-split_*.npy and rewards_split_*.json")
    parser.add_argument("--n_samples_for_fit", type=int, default=10000)
    parser.add_argument("--distill_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of CV folds for isotonic fitting (0=no CV, fit on all)")
    parser.add_argument("--server_alias", type=str, default="lq")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load embeddings and golden scores
    # ------------------------------------------------------------------
    print("Loading data...")
    Embeds, Y_golden = load_local_data(args.data_dir, max_samples=args.n_samples_for_fit)
    if Embeds is None:
        print(f"ERROR: No data found in {args.data_dir}/")
        return
    input_dim = Embeds.shape[-1]
    print(f"Loaded {len(Embeds)} samples, embedding dim = {input_dim}")

    # ------------------------------------------------------------------
    # 2. Split: calibration (60%) / student-train (20%) / held-out (20%)
    #    ISO fit on calibration, student trained on student-train,
    #    student-isotonic fit on calibration predictions.
    # ------------------------------------------------------------------
    idx = np.arange(len(Embeds))
    np.random.seed(42)
    np.random.shuffle(idx)
    n_cal = int(len(idx) * 0.6)
    n_str = int(len(idx) * 0.2)
    cal_idx = idx[:n_cal]
    str_idx = idx[n_cal:n_cal+n_str]
    val_idx = idx[n_cal+n_str:]

    E_cal, Y_cal = Embeds[cal_idx], Y_golden[cal_idx]
    E_str, Y_str = Embeds[str_idx], Y_golden[str_idx]
    E_val, Y_val = Embeds[val_idx], Y_golden[val_idx]
    print(f"Data split: calibration={len(E_cal)}, student-train={len(E_str)}, held-out={len(E_val)}")

    # ------------------------------------------------------------------
    # 3. Load Teacher ORM & collect predictions
    # ------------------------------------------------------------------
    print(f"\nLoading teacher model from {args.teacher_ckpt}")
    teacher_model = MLP(input_dim).to(device)
    teacher_model.load_state_dict(
        torch.load(args.teacher_ckpt, map_location=device, weights_only=True)
    )
    teacher_model.eval()

    print("Collecting teacher predictions...")
    T_cal = predict_batch(teacher_model, E_cal, device)
    T_str = predict_batch(teacher_model, E_str, device)
    T_val = predict_batch(teacher_model, E_val, device)

    # Save raw predictions
    np.save(os.path.join(args.output_dir, "predictions_teacher.npy"), T_cal)
    np.save(os.path.join(args.output_dir, "predictions_golden.npy"), Y_cal)

    # ------------------------------------------------------------------
    # 4. Fit Teacher Isotonic Regression (cross-validated to avoid overfit)
    # ------------------------------------------------------------------
    if args.cv_folds > 1:
        print(f"\nFitting cross-validated isotonic regression ({args.cv_folds} folds)...")
        oof_calibrated, ir_teacher = cv_isotonic_fit(
            T_cal, Y_cal, n_folds=args.cv_folds, seed=42
        )
    else:
        print("\nFitting isotonic regression (full data)...")
        ir_teacher = IsotonicRegression(out_of_bounds="clip")
        oof_calibrated = ir_teacher.fit_transform(T_cal, Y_cal)

    # Calibrate student-train and held-out sets with the teacher isotonic
    T_str_iso = ir_teacher.transform(T_str)
    T_val_iso = ir_teacher.transform(T_val)

    np.save(os.path.join(args.output_dir, "predictions_teacher_isotonic.npy"), oof_calibrated)
    joblib.dump(ir_teacher, os.path.join(args.output_dir, "isotonic_teacher_map.joblib"))
    # Also save with legacy name for backward compatibility
    joblib.dump(ir_teacher, os.path.join(args.output_dir, "isotonic_map.joblib"))

    # Quick diagnostic
    sp_teacher = spearmanr(T_val, Y_val).correlation
    sp_teacher_iso = spearmanr(T_val_iso, Y_val).correlation
    print(f"  Teacher Spearman on held-out:          {sp_teacher:.4f}")
    print(f"  Teacher+Isotonic Spearman on held-out: {sp_teacher_iso:.4f}")

    # ------------------------------------------------------------------
    # 5. Train Student (distilled from isotonic-calibrated teacher)
    #    Target = isotonic(teacher(x)), so student learns calibrated scores.
    # ------------------------------------------------------------------
    print(f"\nTraining Student (target = isotonic teacher scores)...")

    # Student trains on student-train set, validates on held-out
    student_model = train_student(
        MLP(input_dim), E_str, T_str_iso, E_val, T_val_iso,
        epochs=args.distill_epochs, lr=args.learning_rate,
        batch_size=1024, device=device, label="Student"
    )

    save_path = os.path.join(args.output_dir, f"distilled_rm_{args.task}_{args.rm_objective}.ckpt")
    torch.save(student_model.state_dict(), save_path)
    print(f"  Student saved to {save_path}")

    # ------------------------------------------------------------------
    # 6. Fit Student Isotonic Regression
    #    Map student(x) → golden score on calibration set.
    #    This gives us the "Student + Isotonic" variant.
    # ------------------------------------------------------------------
    print(f"\nFitting Student Isotonic Regression...")
    S_cal = predict_batch(student_model, E_cal, device)

    if args.cv_folds > 1:
        _, ir_student = cv_isotonic_fit(S_cal, Y_cal, n_folds=args.cv_folds, seed=42)
    else:
        ir_student = IsotonicRegression(out_of_bounds="clip")
        ir_student.fit(S_cal, Y_cal)

    joblib.dump(ir_student, os.path.join(args.output_dir, "isotonic_student_map.joblib"))
    print(f"  Student isotonic map saved.")

    # ------------------------------------------------------------------
    # 7. Evaluate all four variants on held-out set
    # ------------------------------------------------------------------
    S_val = predict_batch(student_model, E_val, device)
    S_val_iso = ir_student.transform(S_val)

    mse = lambda a, b: float(np.mean((a - b) ** 2))
    sp = lambda a, b: float(spearmanr(a, b).correlation)

    results_table = {
        "Teacher ORM":         {"spearman": sp(T_val, Y_val),     "mse": mse(T_val, Y_val)},
        "Teacher + Isotonic":  {"spearman": sp(T_val_iso, Y_val), "mse": mse(T_val_iso, Y_val)},
        "Student":             {"spearman": sp(S_val, Y_val),     "mse": mse(S_val, Y_val)},
        "Student + Isotonic":  {"spearman": sp(S_val_iso, Y_val), "mse": mse(S_val_iso, Y_val)},
    }

    print(f"\n{'='*65}")
    print(f"Held-out Results ({len(E_val)} samples)")
    print(f"{'='*65}")
    print(f"{'Model':<25} {'Spearman':>10} {'MSE':>10}")
    print(f"{'-'*50}")
    for name, m in results_table.items():
        print(f"{name:<25} {m['spearman']:>10.4f} {m['mse']:>10.4f}")

    n_show = min(5, len(E_val))
    print(f"\nScale Check (first {n_show} held-out samples):")
    print(f"{'GoldenTruth':>12} {'Teacher':>12} {'T+Isotonic':>12} {'Student':>12} {'S+Isotonic':>12}")
    for i in range(n_show):
        print(f"{Y_val[i]:>12.4f} {T_val[i]:>12.4f} {T_val_iso[i]:>12.4f} "
              f"{S_val[i]:>12.4f} {S_val_iso[i]:>12.4f}")

    # ------------------------------------------------------------------
    # 8. Save everything
    # ------------------------------------------------------------------
    np.save(os.path.join(args.output_dir, "predictions_student.npy"), S_val)

    config = {
        "embed_model_name": args.embed_model_name,
        "task": args.task,
        "rm_objective": args.rm_objective,
        "teacher_ckpt": args.teacher_ckpt,
        "n_samples": len(Embeds),
        "input_dim": input_dim,
        "distill_epochs": args.distill_epochs,
        "cv_folds": args.cv_folds,
        "held_out_results": results_table,
    }
    with open(os.path.join(args.output_dir, "distillation_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nArtifacts saved to {args.output_dir}/:")
    print(f"  distilled_rm_{args.task}_{args.rm_objective}.ckpt  (student model)")
    print(f"  isotonic_teacher_map.joblib  (teacher → golden isotonic)")
    print(f"  isotonic_student_map.joblib  (student → golden isotonic)")
    print(f"  isotonic_map.joblib          (legacy alias for teacher isotonic)")


if __name__ == "__main__":
    main()
