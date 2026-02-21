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
parser.add_argument("--n_sample", type=int, default=100,
                    help="Number of prompts to use for pairing. Must be <= prompts in data.")
parser.add_argument("--n_pairs", type=int, default=1)
parser.add_argument("--training_epochs", type=int, default=30)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--replacement", type=str, default="replacement_false",
                    choices=["replacement_true", "replacement_false"])
parser.add_argument("--seed", type=int, default=6)
parser.add_argument("--annotation_quality", type=float, default=10)
args = parser.parse_args()
args.sft_obj_suffix = "" if args.sft_obj == "none" else "-gpt4"
args_replacement = True if args.replacement == "replacement_true" else False

np.random.seed(args.seed)
args.rm_name = "rmmistral7b" if "helpful" in args.task else "rayharmless"
os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading: supports local .npy/.json format AND original cluster .pt paths
# ---------------------------------------------------------------------------

def load_local_data(data_dir):
    """Load embeddings and rewards from local .npy/.json files."""
    emb_files = sorted(glob.glob(os.path.join(data_dir, "EMBD-TRAIN-split_*.npy")))
    rew_files = sorted(glob.glob(os.path.join(data_dir, "rewards_split_*.json")))
    if not emb_files or not rew_files:
        return None, None, None

    embedding_dataset = []
    scores_dataset = []
    query_dataset = []
    for emb_f, rew_f in zip(emb_files, rew_files):
        emb = np.load(emb_f)
        embedding_dataset.append(emb)
        rewards = []
        with open(rew_f) as f:
            for line in f:
                data = json.loads(line)
                rewards.append(data["reward"])
        scores_dataset.append(rewards)
        query_dataset.append(list(range(len(rewards))))
    return embedding_dataset, scores_dataset, query_dataset


def load_cluster_data(args):
    """Load from original cluster paths (requires cluster mount)."""
    from datasets import load_dataset
    embedding_dataset, scores_dataset, query_dataset = [], [], []
    for i in range(10):
        emb_path = (
            f"/mnt/bn/hsunvolume-{args.server_alias}/Step4_2_reorg_tensors_all/"
            f"embed_model_name_{args.embed_model_name}_dataset_orig_hh-rlhf-"
            f"{args.task}{args.sft_obj_suffix}_gen_pref_model_name_"
            f"{args.gen_pref_model_name}_golden_rm_name_{args.rm_name}_"
            f"train_test_train_server_alias_lq_embedding_only_rank_{i}.pt"
        )
        embedding_dataset.append(
            torch.load(emb_path, map_location="cpu", weights_only=True).numpy()
        )
        if "helpful" in args.task:
            ds_path = (f"data_10split/rmmistral7b_{args.gen_pref_model_name}_train_"
                       f"hh-rlhf-{args.task}{args.sft_obj_suffix}_database_split{i}.json")
        else:
            ds_path = (f"data_10split/rayharmless_{args.gen_pref_model_name}_train_"
                       f"hh-rlhf-{args.task}{args.sft_obj_suffix}_database_split{i}.json")
        dataset = load_dataset("json", data_files=ds_path, split="train")
        scores_dataset.append(dataset["scores_sorted"])
        query_dataset.append(dataset["query"])
    return embedding_dataset, scores_dataset, query_dataset


# Try local data first, then cluster
embedding_dataset, scores_dataset, query_dataset = load_local_data(args.data_dir)
if embedding_dataset is None:
    print("Local data not found, trying cluster paths...")
    embedding_dataset, scores_dataset, query_dataset = load_cluster_data(args)
else:
    print(f"Loaded {len(embedding_dataset)} splits from {args.data_dir}/")

n_splits = len(embedding_dataset)
n_prompts = len(embedding_dataset[0])

if args.n_sample > n_prompts:
    print(f"Warning: n_sample ({args.n_sample}) > available prompts ({n_prompts}). Clamping.")
    args.n_sample = n_prompts

for i in range(n_splits):
    print(f"Split {i}: embedding shape = {np.shape(embedding_dataset[i])}")

print("shape of embedding:", np.shape(embedding_dataset))
print("shape of scores:", np.shape(scores_dataset))
assert np.shape(embedding_dataset)[1] == np.shape(scores_dataset)[1]

# ---------------------------------------------------------------------------
# Create pairwise comparison data
# ---------------------------------------------------------------------------

positive_sample = []
negative_sample = []
idx_set_list = []
for i in range(args.n_sample):
    if args.consider_first_n > 0:
        idx_set_list.append(
            np.random.choice(n_splits, min(args.consider_first_n, n_splits), replace=False)
        )
    elif args.consider_first_n == -2:
        idx_set_list.append([min(4, n_splits - 1), min(5, n_splits - 1)])
    elif args.consider_first_n == -1:
        idx_set_list.append([0, n_splits - 1])

p_list = []
reward_diff_list = []
rew_scale_factor = 6.0 if "helpful" in args.task else 1.0

for i in range(args.n_sample):
    idx_set_of_prompt_i = np.random.choice(
        idx_set_list[i], args.n_pairs, replace=args_replacement
    )
    idx_j_set = np.random.choice(args.n_sample, args.n_pairs, replace=False)

    for j, idx_j in enumerate(idx_j_set):
        temp_selected_prompt_idx = np.random.choice(idx_set_list[idx_j], 1, replace=False).item()
        if args.annotation_quality < 0:
            if (scores_dataset[temp_selected_prompt_idx][idx_j]
                    > scores_dataset[idx_set_of_prompt_i[j]][i]):
                positive_sample.append(embedding_dataset[temp_selected_prompt_idx][idx_j])
                negative_sample.append(embedding_dataset[idx_set_of_prompt_i[j]][i])
            else:
                positive_sample.append(embedding_dataset[idx_set_of_prompt_i[j]][i])
                negative_sample.append(embedding_dataset[temp_selected_prompt_idx][idx_j])
        else:
            delta_reward = (scores_dataset[temp_selected_prompt_idx][idx_j]
                            - scores_dataset[idx_set_of_prompt_i[j]][i])
            delta_reward = delta_reward / rew_scale_factor
            prob = 1 / (1 + np.exp(-delta_reward * args.annotation_quality))
            if np.random.rand() < prob:
                positive_sample.append(embedding_dataset[temp_selected_prompt_idx][idx_j])
                negative_sample.append(embedding_dataset[idx_set_of_prompt_i[j]][i])
            else:
                positive_sample.append(embedding_dataset[idx_set_of_prompt_i[j]][i])
                negative_sample.append(embedding_dataset[temp_selected_prompt_idx][idx_j])
            p_list.append(prob)
            reward_diff_list.append(delta_reward)

print("mean of p:", np.mean(p_list), "stats", np.abs(np.asarray(p_list) - 0.5).mean())
print("mean of reward_diff:", np.mean(reward_diff_list), "stats", np.abs(reward_diff_list).mean())

p_list_fn = (f"/plist_XPrompt_mlp_{args.rm_objective}_seed{args.seed}_firstn_"
             f"{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_"
             f"replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.npy")
np.save(f"{args.output_dir}" + p_list_fn, p_list)
reward_diff_list_fn = (f"/reward_diff_list_XPrompt_mlp_{args.rm_objective}_seed{args.seed}_firstn_"
                       f"{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_"
                       f"replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.npy")
np.save(f"{args.output_dir}" + reward_diff_list_fn, reward_diff_list)
print(f"p_list saved as: {args.output_dir + p_list_fn}")
print(f"reward_diff_list saved as: {args.output_dir + reward_diff_list_fn}")

# ---------------------------------------------------------------------------
# Prepare tensors
# ---------------------------------------------------------------------------

positive_sample = torch.tensor(np.array(positive_sample))
negative_sample = torch.tensor(np.array(negative_sample))
positive_label = torch.ones(positive_sample.size(0))
negative_label = torch.zeros(negative_sample.size(0))
embedding_dataset_tensor = torch.cat([positive_sample, negative_sample], dim=0)
embedding_labels = torch.cat([positive_label, negative_label], dim=0)

siamese_dataset_positive = torch.cat(
    [positive_sample.unsqueeze(1), negative_sample.unsqueeze(1)], dim=1)
siamese_dataset_negative = torch.cat(
    [negative_sample.unsqueeze(1), positive_sample.unsqueeze(1)], dim=1)
siamese_labels_positive = torch.ones(positive_sample.size(0))
siamese_labels_negative = torch.zeros(negative_sample.size(0))

siamese_dataset = torch.cat([siamese_dataset_positive, siamese_dataset_negative], dim=0)
siamese_labels = torch.cat([siamese_labels_positive, siamese_labels_negative], dim=0)

indices = torch.randperm(embedding_dataset_tensor.size(0))
embedding_dataset_tensor = embedding_dataset_tensor[indices]
embedding_labels = embedding_labels[indices]

indices_siamese = torch.randperm(siamese_dataset.size(0))
siamese_dataset = siamese_dataset[indices_siamese]
siamese_labels = siamese_labels[indices_siamese]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if args.rm_objective == "clf":
    for model_i in range(args.ensemble_number):
        torch.manual_seed(args.seed * 42 + model_i)
        np.random.seed(args.seed * 42 + model_i)
        X_train, X_test, y_train, y_test = train_test_split(
            embedding_dataset_tensor.numpy(), embedding_labels.numpy(),
            test_size=0.2, random_state=42 + model_i)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

        input_dim = X_train_tensor.shape[-1]
        model = MLP(input_dim)
        train_model(model, device, "clf", X_train_tensor, y_train_tensor,
                     X_test_tensor, y_test_tensor,
                     epochs=args.training_epochs, lr=args.learning_rate, batch_size=10240)
        ckpt_name = (f"/XPrompt_mlp_{args.rm_objective}_{model_i}_seed{args.seed}_firstn_"
                     f"{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_"
                     f"replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.ckpt")
        save_model(model, args.output_dir + ckpt_name)
        print(f"Model saved as: {args.output_dir + ckpt_name}")

elif args.rm_objective == "bt":
    for model_i in range(args.ensemble_number):
        torch.manual_seed(args.seed * 42 + model_i)
        np.random.seed(args.seed * 42 + model_i)
        X_train, X_test, y_train, y_test = train_test_split(
            siamese_dataset.numpy(), siamese_labels.numpy(),
            test_size=0.2, random_state=42 + model_i)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

        input_dim = X_train_tensor.shape[-1]
        model = MLP(input_dim)
        train_model(model, device, "siamese", X_train_tensor, y_train_tensor,
                     X_test_tensor, y_test_tensor,
                     epochs=args.training_epochs, lr=args.learning_rate, batch_size=10240)
        ckpt_name = (f"/XPrompt_mlp_{args.rm_objective}_{model_i}_seed{args.seed}_firstn_"
                     f"{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_"
                     f"replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.ckpt")
        save_model(model, args.output_dir + ckpt_name)
        print(f"Model saved as: {args.output_dir + ckpt_name}")
