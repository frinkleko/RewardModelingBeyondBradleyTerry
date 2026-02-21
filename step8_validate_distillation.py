import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import joblib
from scipy.stats import spearmanr
from networks import MLP
from tqdm import tqdm


def evaluate_model(model, embeddings, golden_rewards, is_isotonic=False, ir_model=None):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(embeddings.shape[0]):
            pred = model(embeddings[i].unsqueeze(0)).cpu().item()
            if is_isotonic and ir_model is not None:
                pred = ir_model.transform([pred])[0]
            all_preds.append(pred)

    all_preds = np.array(all_preds)
    corr = spearmanr(all_preds, golden_rewards).correlation
    return all_preds, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_model_name", type=str, default="gemma2b")
    parser.add_argument("--task", type=str, default="helpful")
    parser.add_argument("--sft_obj", type=str, default="gpt4")
    parser.add_argument("--gen_pref_model_name", type=str, default="gemma2b")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--student_ckpt", type=str, required=True)
    parser.add_argument("--isotonic_model", type=str, required=True)
    parser.add_argument("--server_alias", type=str, default="lq")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Models
    # Get input_dim from a sample test embedding
    test_emb_path = f"/mnt/bn/hsunvolume-lq/Step4_2_testall/EXPembed_model_name_{args.embed_model_name}_dataset_orig_hh-rlhf-{args.task}-{args.sft_obj}_gen_pref_model_name_{args.gen_pref_model_name}_n_samples_500_max_len_128_split_0_server_alias_lq/{args.embed_model_name}_{args.gen_pref_model_name}_hh-rlhf-{args.task}-{args.sft_obj}_0_testembedding_total_128.pt"

    if not os.path.exists(test_emb_path):
        print(f"Error: Test embeddings not found at {test_emb_path}")
        return

    test_embeddings_all = torch.load(test_emb_path)  # [500, N_PROMPTS, DIM] or similar
    input_dim = test_embeddings_all.shape[-1]

    teacher = MLP(input_dim).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_to_local=device))

    student = MLP(input_dim).to(device)
    student.load_state_dict(torch.load(args.student_ckpt, map_to_local=device))

    ir_model = joblib.load(args.isotonic_model)

    # 2. Load Test Ground Truth
    rm_name = "rmmistral7b" if args.task == "helpful" else "rayharmless"
    gt_file = f"/mnt/bn/hsunvolume-lq/Step3_Annotate_Test_RPT/EXPmodel_name_{args.gen_pref_model_name}_dataset_hh-rlhf-{args.task}-{args.sft_obj}_n_samples_500_split_0_data_class_test_server_alias_yg_rm{rm_name}_0_maxlen128.json"

    reward_list = []
    with open(gt_file, "r") as f:
        for line in f:
            data = json.loads(line)
            sub_rewards = [data[f"rm_{rm_name}_{idx}"] for idx in range(500)]
            reward_list.append(sub_rewards)

    # 3. Evaluation
    print("Evaluating models on test set...")
    # reward_list shape: [N_PROMPTS, 500]
    # test_embeddings_all shape: [500, N_PROMPTS, DIM]

    n_prompts = len(reward_list)
    teacher_corrs = []
    iso_corrs = []
    student_corrs = []

    for p_idx in tqdm(range(n_prompts)):
        embs = test_embeddings_all[:, p_idx, :].to(device)  # [500, DIM]
        gt = np.array(reward_list[p_idx])

        _, t_corr = evaluate_model(teacher, embs, gt)
        _, i_corr = evaluate_model(
            teacher, embs, gt, is_isotonic=True, ir_model=ir_model
        )
        _, s_corr = evaluate_model(student, embs, gt)

        teacher_corrs.append(t_corr)
        iso_corrs.append(i_corr)
        student_corrs.append(s_corr)

    print(f"Results (Mean Spearman Correlation):")
    print(f"Teacher ORM:         {np.nanmean(teacher_corrs):.4f}")
    print(
        f"Teacher + Isotonic:  {np.nanmean(iso_corrs):.4f} (Should be same as Teacher)"
    )
    print(f"Student (Distilled): {np.nanmean(student_corrs):.4f}")

    # 4. Calibration Check (MSE on test set)
    # Pick a few samples to see the scale
    print("Scale Check (First 5 samples of first prompt):")
    embs = test_embeddings_all[:5, 0, :].to(device)
    gt = np.array(reward_list[0][:5])
    with torch.no_grad():
        t_out = teacher(embs).cpu().numpy().flatten()
        i_out = ir_model.transform(t_out)
        s_out = student(embs).cpu().numpy().flatten()

    print(f"Ground Truth: {gt}")
    print(f"Teacher Raw:  {t_out}")
    print(f"Isotonic Map: {i_out}")
    print(f"Student RM:   {s_out}")


if __name__ == "__main__":
    main()
