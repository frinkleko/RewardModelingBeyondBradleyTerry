import os
from tqdm import tqdm
import json
import numpy as np
import argparse
from datasets import Dataset, load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="distill_out",
                    help="Base output directory (same as steps 1-3)")
parser.add_argument("--model_name", type=str, default="gemma2b")
parser.add_argument("--dataset", type=str, default="hh-rlhf-helpful-gpt4")
parser.add_argument("--eval_dataset", type=str, default="hh-rlhf-helpful")
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()

output_dir = args.output_dir
temp_out = os.path.join(output_dir, "temp_out_data")
os.makedirs(temp_out, exist_ok=True)

model_name = args.model_name
dataset_name = args.dataset

if "helpful" in dataset_name:
    reward_model = "rmmistral7b"
elif "harmless" in dataset_name:
    reward_model = "rayharmless"
else:
    reward_model = None

for data_cls in ["train", "test"]:
    if data_cls == "test":
        n_samples = 500
    elif data_cls == "train":
        n_samples = 10

    print("current setups:", model_name, data_cls, dataset_name)
    json_out_line = [{} for _ in range(n_samples)]
    for split in range(5):
        # Build path matching step3's output naming convention
        # Step3's response_path has trailing /, and writes _rm... appended to it
        folder_name = (
            f"{output_dir}/Part_{split}_sft_sftmax_len{args.max_len}"
            f"_temp{args.temperature}_{model_name}_{dataset_name}"
            f"_{args.eval_dataset}_n{n_samples}_dcls{data_cls}/"
        )

        json_file = f"{folder_name}_rm{reward_model}_{split}_maxlen{args.max_len}.json"
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} not found, skipping.")
            continue
        out_data = []
        with open(json_file) as f:
            for line in f:
                config = json.loads(line)
                out_data.append(config)

        for line in tqdm(out_data):
            for i in range(n_samples):
                json_out_line[i]["query"] = line["query"]
            gen_scores = []
            gen_responses_trunc = []
            for sample_i in range(n_samples):
                gen_scores.append(line[f"rm_{reward_model}_{sample_i}"])
                gen_responses_trunc.append(line[f"trunc_response_{sample_i}"])
            # sort scores and responses
            gen_scores = np.array(gen_scores)
            gen_responses_trunc = np.array(gen_responses_trunc)
            gen_scores_sorted_idx = np.argsort(gen_scores)
            gen_scores_sorted = gen_scores[gen_scores_sorted_idx]
            gen_responses_trunc_sorted = gen_responses_trunc[
                gen_scores_sorted_idx
            ]

            for i in range(n_samples):
                json_out_line[i]["scores_sorted"] = gen_scores_sorted[i]
                json_out_line[i]["responses_sorted"] = (
                    gen_responses_trunc_sorted[i]
                )

                out_file = os.path.join(
                    temp_out,
                    f"{reward_model}_{model_name}_{data_cls}_{args.eval_dataset}_database_split{i}.json",
                )
                with open(out_file, "a+") as f:
                    json.dump(json_out_line[i], f)
                    f.write("\n")
