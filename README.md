# Rethinking Bradley-Terry Models in Preference-Based Reward Modeling

> **ICLR 2025** — [Paper](https://openreview.net/forum?id=rfdblE10qm) · [Website](https://sites.google.com/view/rewardmodels) · [Preprint](https://arxiv.org/pdf/2411.04991) · [Embedding Infrastructure](https://github.com/holarissun/embedding-based-llm-alignment)

![Paper Preview](img/paper_prev.png)

---

## Why This Repo Exists

Standard RLHF reward models learn **relative preferences** (Bradley-Terry: "A is better than B") but never produce **calibrated scalar rewards** needed by PPO.  This repo implements the full pipeline from the ICLR 2025 paper to:

1. Train **Ordinal Reward Models (ORMs)** on embedding-space pairwise comparisons.
2. **Calibrate** ORM outputs to golden reward scores via isotonic regression.
3. **Distill** the calibrated mapping into a single fast MLP student.

The key insight is that ORMs trained with Bradley-Terry loss learn a latent ordering of responses, but the raw logits are on an **arbitrary scale**. Isotonic regression provides a monotone mapping from ORM logits to golden reward values, preserving rank while recovering meaningful magnitudes.

### What the Four Model Variants Mean

| Variant | Inference | What it does |
|---|---|---|
| **Teacher ORM** | `model(embedding)` -> raw logit | Ranks well, but output scale is arbitrary |
| **Teacher + Isotonic** | `isotonic.transform(model(embedding))` | Maps logits to golden scale, preserving rank |
| **Student** | `student(embedding)` -> scalar | Single MLP trained to mimic isotonic-calibrated scores |
| **Student + Isotonic** | `isotonic_student.transform(student(embedding))` | Further calibrates the student for best MSE |

**Evaluation note:** MSE of the raw Teacher ORM vs golden scores is **not meaningful** -- the ORM is trained for ranking (pairwise sigmoid), not regression. The relevant metrics are **Spearman correlation** and **pairwise accuracy**. MSE only becomes meaningful after isotonic calibration maps predictions to the golden reward scale.

---

_This is Part I of a series on embedding-based reward models in RLHF:_
- **Part I** -- Reward Model Foundations (this paper)
- **Part II** -- Active Reward Modeling ([preprint](https://arxiv.org/abs/2502.04354), [repo](https://github.com/YunyiShen/ARM-FI))
- **Part III** -- Embedding Infrastructure ([preprint](https://arxiv.org/pdf/2502.04357), [repo](https://github.com/holarissun/embedding-based-llm-alignment))
- **Part IV** -- Human Preference via PCA ([preprint](https://arxiv.org/pdf/2502.13131))

---

## Quick Start

### Prerequisites

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch numpy scikit-learn scipy tqdm joblib datasets transformers
```

### Option A: Run with Demo Data (CPU, ~2 min)

Generates synthetic embeddings and runs the full pipeline to verify everything works:

```bash
bash run_all.sh
```

### Option B: Run with Real Embeddings (CPU, ~5 min)

Download pre-computed embeddings from [embedding-based-llm-alignment](https://github.com/holarissun/embedding-based-llm-alignment):

```bash
# 1. Download embedding files (see "Downloading Embeddings" below)
# 2. Run pipeline
USE_REPO_EMBD=1 TASK=harmless N_SAMPLE=1000 ENSEMBLE_NUMBER=3 \
    GEN_PREF_MODEL=gemma2b MAX_PROMPTS=5000 bash run_all.sh
```

### Option C: Generate Everything from Scratch (GPU required)

```bash
RUN_GPU_STEPS=1 bash run_all.sh
```

---

## Downloading Embeddings

### Single-Experiment Data (~10 GB)
From the [embedding-based-llm-alignment repo](https://github.com/holarissun/embedding-based-llm-alignment), download individual files from Google Drive.  Place them under `embd/RM-Embeddings/`:

```
embd/RM-Embeddings/
  EMBD_hh-rlhf-harmless-gpt4_gemma2b_gemma7b_train.npy   # (10, N, 2048)
  REWARD_hh-rlhf-harmless-gpt4_rayharmless_gemma2b_train.npy  # (10, N)
  REWARD_hh-rlhf-harmless-gpt4_rayharmless_gemma7b_train.npy
  ...
```

### Full Dataset (~300 GB)
The complete dataset covers all model/task/SFT combinations:
- **Models:** gemma-2b, gemma-7b, llama3-8b
- **Tasks:** helpful, harmless
- **SFT objectives:** gpt4, none

Available at: https://drive.google.com/drive/folders/1cRiwvZDxlq_5DVHBIIVYjeunse42ALMO

To use a specific combination:
```bash
USE_REPO_EMBD=1 TASK=helpful GEN_PREF_MODEL=gemma7b \
    REPO_GEN_MODEL=gemma7b N_SAMPLE=5000 bash run_all.sh
```

### Data Format

**Repository format** (as downloaded):
- Embeddings: `(K, N, D)` -- K=10 responses per prompt, N prompts, D=2048 embedding dim
- Rewards: `(K, N)` -- golden reward scores from calibrated reward models

**Local pipeline format** (after conversion by `convert_embeddings.py`):
- `EMBD-TRAIN-split_{k}.npy`: shape `(N, D)` -- one split per response index
- `rewards_split_{k}.json`: JSONL with `{"reward": float}` per prompt

---

## Pipeline Overview

```
Steps 1-4: Data Generation (GPU, optional)

  step1_sft.py           ->  Fine-tune base LLM with LoRA
  step2_gen_sample.py    ->  Generate responses via vLLM
  step3_reward_ann.py    ->  Score responses with golden RM
  step3.5_processing.py  ->  Re-organize into sorted splits
  step4_gen_embeddings   ->  Extract LLM hidden-state embeddings
           |
           | produces: EMBD-TRAIN-split_k.npy + rewards_split_k.json
           v
Steps 5-8: Reward Model Research (CPU-only)

  step5_train_rms.py     ->  Train ORM ensemble (BT or classification)
  step6_eval_rms.py      ->  Evaluate ranking quality (Spearman, acc)
  step7_isotonic_dist.py ->  Isotonic calibration + student distillation
  step8_validate_dist.py ->  Compare all 4 variants on held-out data
```

---

## Design: ORM -> Isotonic -> Student

### Why ORM Scores Need Calibration

A Bradley-Terry ORM is trained with the loss:

    L_BT = -log sigma(r(x, y_w) - r(x, y_l))

This teaches the MLP that `model(better_response) > model(worse_response)`, but the raw outputs live on an **arbitrary scale** determined by the training dynamics.  For example:

| Response | Golden Reward | ORM Raw Output |
|---|---|---|
| Good | +1.59 | -0.36 |
| Bad  | -2.94 | -2.01 |

The ORM correctly ranks them (-0.36 > -2.01), but the values are not on the golden reward scale.  You cannot use these raw logits as PPO rewards -- the magnitude, offset, and nonlinear scaling are all wrong.

**MSE between raw ORM outputs and golden scores is therefore not meaningful** -- it compares two fundamentally different scales.  The proper evaluation metric for the raw ORM is **Spearman correlation** or **pairwise accuracy**, which measure ranking quality only.

### Why Isotonic Regression

**Isotonic regression** fits a monotonically non-decreasing function f that minimizes:

    min_f  sum_i (f(r_hat_i) - r*_i)^2
    s.t.   r_hat_i <= r_hat_j  =>  f(r_hat_i) <= f(r_hat_j)

where r_hat_i = ORM prediction, r*_i = golden reward.

This is ideal because:
1. **Preserves ranking** -- the monotonicity constraint guarantees rank order is maintained.
2. **Nonparametric** -- makes no assumption about the ORM's output distribution.
3. **Recovers scale** -- maps to the golden reward range, making MSE meaningful.
4. **Fast** -- O(N) via pool-adjacent-violators algorithm.

### Why Distill into a Student

The Teacher+Isotonic pipeline requires two inference steps (MLP then lookup table).  The **Student MLP** learns to directly predict `isotonic(teacher(x))` from embeddings in a single forward pass, which is:
- Simpler to deploy (one `.ckpt` file)
- Compatible with standard PPO reward interfaces
- Allows further calibration via Student+Isotonic

### Cross-Validated Isotonic (Avoiding Overfitting)

Plain isotonic regression fit on training data **overfits** -- it memorizes a piecewise-constant function that degenerates to near-constant outputs on unseen data.  We use **5-fold cross-validation**: fit on K-1 folds, predict on the held-out fold.  This produces honest out-of-fold calibrated targets for student training, and the final isotonic model is refit on all data for inference.

### Data Splitting Strategy

```
All data (N samples)
  60% Calibration set    ->  fit isotonic (teacher->golden), fit isotonic (student->golden)
  20% Student-train set  ->  train student MLP on isotonic-calibrated targets
  20% Held-out set       ->  evaluate all 4 variants (never seen during fitting)
```

---

## run_all.sh Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `RUN_GPU_STEPS` | `0` | Set `1` to run steps 1-4 (requires GPU) |
| `USE_REPO_EMBD` | `0` | Set `1` to use downloaded embedding repo data |
| `TASK` | `helpful` | Task: `helpful` or `harmless` |
| `MODEL_NAME` | `gemma2b` | Embedding model |
| `GEN_PREF_MODEL` | `gemma7b` | Generation model for pref data |
| `N_SAMPLE` | `100` | Number of prompts for ORM training pairs |
| `ENSEMBLE_NUMBER` | `10` | Number of ORM ensemble members |
| `TRAINING_EPOCHS` | `2` | Epochs for ORM training |
| `DISTILL_EPOCHS` | `20` | Epochs for student distillation |
| `RM_OBJECTIVE` | `bt` | `bt` (Bradley-Terry) or `clf` (classification) |
| `ANNOTATION_QUALITY` | `10` | Higher = less label noise in pairwise data |
| `OUTPUT_DIR` | `distill_out` | Output directory |
| `DATA_DIR` | `data` | Embedding data directory |
| `MAX_PROMPTS` | (all) | Limit prompts when converting repo embeddings |
| `EMBD_REPO_DIR` | `embd/RM-Embeddings` | Path to downloaded repo data |

### Examples

```bash
# Quick demo test
bash run_all.sh

# Real harmless data, 3 ensemble models, 1000 prompts
USE_REPO_EMBD=1 TASK=harmless N_SAMPLE=1000 ENSEMBLE_NUMBER=3 \
    GEN_PREF_MODEL=gemma2b MAX_PROMPTS=5000 bash run_all.sh

# Real helpful data with gemma7b generation model
USE_REPO_EMBD=1 TASK=helpful GEN_PREF_MODEL=gemma7b \
    N_SAMPLE=2000 ENSEMBLE_NUMBER=5 TRAINING_EPOCHS=10 bash run_all.sh

# Full GPU pipeline
RUN_GPU_STEPS=1 MODEL_NAME=gemma2b DATASET=hh-rlhf-helpful-gpt4 bash run_all.sh
```

---

## Step-by-Step Reference

### Steps 1-4: GPU Data Generation

```bash
# Step 1: SFT with LoRA
python3 step1_sft.py --model_name gemma2b --dataset hh-rlhf-helpful-gpt4

# Step 2: Generate responses (10 per prompt for train, 500 for test)
python3 step2_gen_sample.py --model_name gemma2b --adapter_name sft \
    --dataset hh-rlhf-helpful-gpt4 --eval_dataset hh-rlhf-helpful \
    --data_class train --n_samples 10 --max_len 128

# Step 3: Annotate with golden reward model
python3 step3_reward_annotation.py --adapter_name sft --model_name gemma2b \
    --dataset hh-rlhf-helpful-gpt4 --eval_dataset hh-rlhf-helpful \
    --data_class train --n_samples 10

# Step 3.5: Re-organize scored data
python3 step3.5_processing_data.py --output_dir distill_out \
    --model_name gemma2b --dataset hh-rlhf-helpful-gpt4 \
    --eval_dataset hh-rlhf-helpful

# Step 4: Extract embeddings
python3 step4_gen_embeddings.py --embed_model_name gemma2b \
    --dataset hh-rlhf-helpful-gpt4 --gen_pref_model_name gemma2b \
    --train_test train --n_samples 10
```

### Steps 5-8: CPU Reward Model Research

```bash
# Step 5: Train ORM ensemble
python3 step5_train_rms.py --embed_model_name gemma2b --task helpful \
    --sft_obj gpt4 --gen_pref_model_name gemma2b --rm_objective bt \
    --consider_first_n 2 --n_sample 1000 --training_epochs 5 \
    --ensemble_number 3

# Step 6: Evaluate ORM ranking quality
python3 step6_eval_rms.py --embed_model_name gemma2b --task helpful \
    --sft_obj gpt4 --gen_pref_model_name gemma2b --rm_objective bt \
    --consider_first_n 2 --n_sample 1000 --training_epochs 5 \
    --ensemble_number 3

# Step 7: Isotonic calibration + student distillation
python3 step7_isotonic_distillation.py --task helpful --rm_objective bt \
    --teacher_ckpt distill_out/XPrompt_mlp_bt_2_seed6_...ckpt \
    --distill_epochs 30

# Step 8: Validate all 4 variants
python3 step8_validate_distillation.py --task helpful \
    --teacher_ckpt distill_out/XPrompt_mlp_bt_2_seed6_...ckpt \
    --student_ckpt distill_out/distilled_rm_helpful_bt.ckpt \
    --isotonic_model distill_out/isotonic_map.joblib
```

### Key step5 Parameters

| Parameter | Values | Effect |
|---|---|---|
| `--rm_objective` | `bt` / `clf` | Bradley-Terry (pairwise) vs classification |
| `--consider_first_n` | `2` / `-1` / `-2` | `2`: random 2 of 10 responses; `-1`: max diversity; `-2`: min diversity |
| `--annotation_quality` | `10` / `1.0` / `0.5` | Higher = less noisy labels (10 < 5% error, 0.5 ~ 40% error) |
| `--n_sample` | int | Number of prompts used for pair construction |

---

## Scaling Up with the Full 300 GB Dataset

The [full embedding dataset](https://drive.google.com/drive/folders/1cRiwvZDxlq_5DVHBIIVYjeunse42ALMO) covers all combinations of:

| Dimension | Options |
|---|---|
| Embedding model | gemma-2b, gemma-7b, llama3-8b |
| Task | helpful, harmless |
| SFT objective | gpt4, none |
| Golden RM | rmmistral7b (helpful), rayharmless (harmless) |

### Recommended Experiments with Full Data

```bash
# 1. Cross-model comparison: same task, different embedding models
for MODEL in gemma2b gemma7b llama38b; do
    USE_REPO_EMBD=1 TASK=helpful MODEL_NAME=$MODEL GEN_PREF_MODEL=$MODEL \
        REPO_GEN_MODEL=$MODEL N_SAMPLE=5000 ENSEMBLE_NUMBER=10 \
        OUTPUT_DIR=out_${MODEL}_helpful bash run_all.sh
done

# 2. Annotation quality ablation
for AQ in 10 1.0 0.5 0.1; do
    USE_REPO_EMBD=1 TASK=helpful N_SAMPLE=5000 ANNOTATION_QUALITY=$AQ \
        OUTPUT_DIR=out_aq_${AQ} bash run_all.sh
done

# 3. Sample efficiency: vary number of training prompts
for NS in 100 500 1000 5000 10000; do
    USE_REPO_EMBD=1 TASK=helpful N_SAMPLE=$NS \
        OUTPUT_DIR=out_n_${NS} bash run_all.sh
done

# 4. BT vs Classification objective
for OBJ in bt clf; do
    USE_REPO_EMBD=1 TASK=helpful RM_OBJECTIVE=$OBJ \
        OUTPUT_DIR=out_${OBJ} bash run_all.sh
done
```

### What to Measure

- **Pairwise accuracy**: fraction of response pairs correctly ranked (primary metric)
- **Spearman rho**: rank correlation with golden reward (global ranking quality)
- **MSE** (only after isotonic): distance to golden reward on calibrated scale
- Compare across: model sizes, tasks, training set sizes, annotation noise levels

---

## Output Artifacts

After a pipeline run, `distill_out/` contains:

| File | Description |
|---|---|
| `XPrompt_mlp_bt_*_seed*.ckpt` | Trained ORM ensemble checkpoints |
| `distilled_rm_{task}_{obj}.ckpt` | Student MLP checkpoint |
| `isotonic_teacher_map.joblib` | Teacher to golden isotonic mapping |
| `isotonic_student_map.joblib` | Student to golden isotonic mapping |
| `isotonic_map.joblib` | Legacy alias for teacher isotonic |
| `eval_results_*.json` | Step 6 evaluation results |
| `validation_results.json` | Step 8 four-variant comparison |
| `distillation_config.json` | Distillation hyperparameters + results |

---

## File Structure

```
run_all.sh                      # Main pipeline script
networks.py                     # MLP architecture, BT/CLF training
step1_sft.py                    # GPU: LoRA fine-tuning
step2_gen_sample.py             # GPU: vLLM response generation
step3_reward_annotation.py      # GPU: golden RM scoring
step3.5_processing_data.py      # GPU: data reorganization
step4_gen_embeddings.py         # GPU: embedding extraction
step5_train_rms.py              # CPU: ORM ensemble training
step6_eval_rms.py               # CPU: ORM evaluation
step7_isotonic_distillation.py  # CPU: isotonic calibration + distillation
step8_validate_distillation.py  # CPU: 4-variant validation
convert_embeddings.py           # Convert repo embedding format to local
generate_demo_data.py           # Generate synthetic test data
data/                           # Embedding data directory
```

---

## Call for Contribution

We welcome contributors to expand the embedding dataset! If you have embeddings or golden-reward annotations from your reward model research, please contact `sunhopht@gmail.com`.

## Citation

```bibtex
@inproceedings{
  sun2025rethinking,
  title={Rethinking Reward Modeling in Preference-based Large Language Model Alignment},
  author={Hao Sun and Yunyi Shen and Jean-Francois Ton},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=rfdblE10qm}
}
```
