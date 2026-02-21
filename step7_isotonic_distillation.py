import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import json
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from networks import MLP, train_model, save_model
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_model_name", type=str, default="gemma2b")
    parser.add_argument("--task", type=str, default="helpful")
    parser.add_argument("--sft_obj", type=str, default="gpt4")
    parser.add_argument("--gen_pref_model_name", type=str, default="gemma2b")
    parser.add_argument("--rm_objective", type=str, choices=["clf", "bt"], default="bt")
    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        required=True,
        help="Path to the trained ORM checkpoint",
    )
    parser.add_argument("--output_dir", type=str, default="distill_out")
    parser.add_argument("--n_samples_for_fit", type=int, default=10000)
    parser.add_argument("--distill_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--server_alias", type=str, default="lq")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Teacher ORM
    # We need input_dim, which we can get from the checkpoint or by loading one embedding
    # For now, let's assume 2048 or 3584 based on Gemma
    print("Loading teacher model...")
    # Loading first embedding to get dimension
    sample_emb_path = f"/mnt/bn/hsunvolume-{args.server_alias}/Step4_2_reorg_tensors_all/embed_model_name_{args.embed_model_name}_dataset_orig_hh-rlhf-{args.task}-{args.sft_obj}_gen_pref_model_name_{args.gen_pref_model_name}_golden_rm_name_rmmistral7b_train_test_train_server_alias_lq_embedding_only_rank_0.pt"

    # Fallback if path doesn't exist for local testing/dev
    if not os.path.exists(sample_emb_path):
        print(
            f"Warning: Could not find embeddings at {sample_emb_path}. Using dummy dimension 2048 for script structure."
        )
        input_dim = 2048
    else:
        sample_emb = torch.load(sample_emb_path)
        input_dim = sample_emb.shape[-1]

    teacher_model = MLP(input_dim).to(device)
    teacher_model.load_state_dict(torch.load(args.teacher_ckpt, map_to_local=device))
    teacher_model.eval()

    # 2. Collect Calibration Data (MLP Outputs vs Golden Scores)
    print("Collecting calibration data...")
    all_mlp_scores = []
    all_golden_scores = []
    all_embeddings = []

    # Check local data or cluster mount
    cluster_base = f"/mnt/bn/hsunvolume-{args.server_alias}/Step4_2_reorg_tensors_all"
    local_base = "data"

    for i in range(10):
        # Try cluster path then local path
        emb_fn = f"embed_model_name_{args.embed_model_name}_dataset_orig_hh-rlhf-{args.task}-{args.sft_obj}_gen_pref_model_name_{args.gen_pref_model_name}_golden_rm_name_rmmistral7b_train_test_train_server_alias_lq_embedding_only_rank_{i}.pt"
        emb_path = os.path.join(cluster_base, emb_fn)
        if not os.path.exists(emb_path):
            emb_path = os.path.join(local_base, emb_fn)

        if not os.path.exists(emb_path):
            print(f"Skipping split {i}: Embedding file not found at {emb_path}")
            continue

        embeddings = torch.load(emb_path)

        # Check data_10split or temp_out_data or data
        score_fn = f"rmmistral7b_{args.gen_pref_model_name}_train_hh-rlhf-{args.task}-{args.sft_obj}_database_split{i}.json"
        score_file = os.path.join("data_10split", score_fn)
        if not os.path.exists(score_file):
            score_file = os.path.join(
                "temp_out_data", score_file
            )  # check alternate loc
        if not os.path.exists(score_file):
            score_file = os.path.join("data", score_fn)

        if not os.path.exists(score_file):
            print(f"Skipping split {i}: Score file not found at {score_file}")
            continue

        with open(score_file, "r") as f:
            for line_idx, line in enumerate(f):
                data = json.loads(line)
                golden_score = data[
                    "scores_sorted"
                ]  # This is the scalar from the Golden RM

                # Get embedding for this sample
                # In step5, it seems embeddings are [N_PROMPTS, N_SAMPLES_PER_PROMPT, DIM]
                # But Step4 reorg might be different. Let's assume it matches the JSON line index.
                emb = embeddings[line_idx]

                with torch.no_grad():
                    mlp_score = teacher_model(emb.to(device).unsqueeze(0)).cpu().item()

                all_mlp_scores.append(mlp_score)
                all_golden_scores.append(golden_score)
                all_embeddings.append(emb.numpy())

                if len(all_mlp_scores) >= args.n_samples_for_fit:
                    break
        if len(all_mlp_scores) >= args.n_samples_for_fit:
            break

    X_mlp = np.array(all_mlp_scores)
    Y_golden = np.array(all_golden_scores)
    Embeds = np.array(all_embeddings)

    # 3. Fit Isotonic Regression
    print("Fitting Isotonic Regression...")
    ir = IsotonicRegression(out_of_bounds="clip")
    Y_calibrated = ir.fit_transform(X_mlp, Y_golden)

    # 4. Train Student Model (Optional but requested for "explicit reward for training like PPO")
    # This student will learn to map Embedding -> Calibrated Reward directly
    print("Training Student Distilled RM...")
    X_train, X_test, y_train, y_test = train_test_split(
        Embeds, Y_calibrated, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    student_model = MLP(input_dim).to(device)
    criterion = nn.MSELoss()  # Distillation uses MSE for value fitting
    optimizer = optim.Adam(student_model.parameters(), lr=args.learning_rate)

    # Simple training loop for distillation
    batch_size = 1024
    for epoch in range(args.distill_epochs):
        student_model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = student_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        student_model.eval()
        with torch.no_grad():
            val_outputs = student_model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)

        print(
            f"Epoch {epoch+1}/{args.distill_epochs}, Loss: {epoch_loss/(len(X_train)/batch_size):.4f}, Val MSE: {val_loss.item():.4f}"
        )

    # 5. Save Results
    save_path = os.path.join(
        args.output_dir, f"distilled_rm_{args.task}_{args.rm_objective}.ckpt"
    )
    torch.save(student_model.state_dict(), save_path)

    # Save isotonic model for reference
    import joblib

    joblib.dump(ir, os.path.join(args.output_dir, "isotonic_map.joblib"))
    print(f"Distilled model saved to {save_path}")


if __name__ == "__main__":
    main()
