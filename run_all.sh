#!/usr/bin/env bash
# =============================================================================
# run_all.sh  –  Reproduce Steps 1-8 for RewardModelingBeyondBradleyTerry
#
# Uses `uv run` to execute every script inside the project virtual environment.
#
# Steps 1-4: GPU-required (optional, for generating data from scratch)
# Steps 5-8: CPU-friendly (uses pre-computed embeddings in data/)
#
# Usage:
#   bash run_all.sh                        # Run CPU-only steps 5-8 (demo data)
#   RUN_GPU_STEPS=1 bash run_all.sh        # Run ALL steps 1-8 (needs GPU)
#   USE_REPO_EMBD=1 bash run_all.sh        # Use downloaded embeddings from
#                                          # holarissun/embedding-based-llm-alignment
#   USE_REPO_EMBD=1 TASK=harmless bash run_all.sh  # Run with harmless task
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Configuration  (edit these to match your setup)
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-distill_out}"

# Step 1-4 settings (GPU steps)
MODEL_NAME="${MODEL_NAME:-gemma2b}"
DATASET="${DATASET:-hh-rlhf-helpful-gpt4}"
EVAL_DATASET="${EVAL_DATASET:-hh-rlhf-helpful}"

# Step 5-8 settings
TASK="${TASK:-helpful}"
SFT_OBJ="${SFT_OBJ:-gpt4}"
GEN_PREF_MODEL="${GEN_PREF_MODEL:-gemma7b}"
RM_OBJECTIVE="${RM_OBJECTIVE:-bt}"
CONSIDER_FIRST_N="${CONSIDER_FIRST_N:-2}"
N_SAMPLE="${N_SAMPLE:-100}"
TRAINING_EPOCHS="${TRAINING_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SEED="${SEED:-6}"
ANNOTATION_QUALITY="${ANNOTATION_QUALITY:-10}"
ENSEMBLE_NUMBER="${ENSEMBLE_NUMBER:-10}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-20}"
NORMALIZE_REWARDS="${NORMALIZE_REWARDS:-0}"
STUDENT_LOSS="${STUDENT_LOSS:-hybrid}"
RANKING_LAMBDA="${RANKING_LAMBDA:-1.0}"
BINARIZE_REWARDS="${BINARIZE_REWARDS:-0}"

# Derived
REPLACEMENT="replacement_false"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
section() {
    echo ""
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
}

# ---------------------------------------------------------------------------
# Steps 1-4: GPU-required data generation (optional)
# ---------------------------------------------------------------------------
if [[ "${RUN_GPU_STEPS:-0}" == "1" ]]; then

    section "Step 1: Supervised Fine-Tuning (SFT)"
    uv run python3 step1_sft.py \
        --model_name "$MODEL_NAME" \
        --dataset "$DATASET" \
        --learning_rate 5e-5 \
        --epochs 2 \
        --output_dir "$OUTPUT_DIR"

    section "Step 2: Generate Samples (train + test)"
    # Training samples (10 per prompt)
    for SPLIT in 0 1 2 3 4; do
        uv run python3 step2_gen_sample.py \
            --model_name "$MODEL_NAME" \
            --adapter_name sft \
            --dataset "$DATASET" \
            --eval_dataset "$EVAL_DATASET" \
            --data_class train \
            --n_samples 10 \
            --max_len 128 \
            --split "$SPLIT" \
            --output_dir "$OUTPUT_DIR"
    done
    # Test samples (500 per prompt)
    for SPLIT in 0 1 2 3 4; do
        uv run python3 step2_gen_sample.py \
            --model_name "$MODEL_NAME" \
            --adapter_name sft \
            --dataset "$DATASET" \
            --eval_dataset "$EVAL_DATASET" \
            --data_class test \
            --n_samples 500 \
            --max_len 128 \
            --split "$SPLIT" \
            --output_dir "$OUTPUT_DIR"
    done

    section "Step 3: Reward Annotation"
    for SPLIT in 0 1 2 3 4; do
        uv run python3 step3_reward_annotation.py \
            --adapter_name sft \
            --model_name "$MODEL_NAME" \
            --dataset "$DATASET" \
            --eval_dataset "$EVAL_DATASET" \
            --data_class train \
            --n_samples 10 \
            --split "$SPLIT" \
            --output_dir "$OUTPUT_DIR"
    done

    section "Step 3.5: Processing Data"
    uv run python3 step3.5_processing_data.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --dataset "$DATASET" \
        --eval_dataset "$EVAL_DATASET" \
        --max_len 128 \
        --temperature 1.0

    section "Step 4: Generate Embeddings"
    for SPLIT in 0 1 2 3 4; do
        # Step4 reads config.json from step2's output directory
        STEP2_DIR="${OUTPUT_DIR}/Part_${SPLIT}_sft_sftmax_len128_temp1.0_${MODEL_NAME}_${DATASET}_${EVAL_DATASET}_n10_dclstrain"
        uv run python3 step4_gen_embeddings.py \
            --embed_model_name "$MODEL_NAME" \
            --dataset "$DATASET" \
            --gen_pref_model_name "$MODEL_NAME" \
            --train_test train \
            --n_samples 10 \
            --split "$SPLIT" \
            --output_dir "$STEP2_DIR"
    done

    echo ""
    echo "GPU steps (1-4) complete. Data stored in $OUTPUT_DIR/"

else
    echo "Skipping GPU steps 1-4 (set RUN_GPU_STEPS=1 to enable)."

    if [[ "${USE_REPO_EMBD:-0}" == "1" ]]; then
        # Convert downloaded embedding-repo data to local format
        EMBD_REPO_DIR="${EMBD_REPO_DIR:-embd/RM-Embeddings}"
        REPO_GEN_MODEL="${REPO_GEN_MODEL:-gemma2b}"     # generation model in repo
        MAX_PROMPTS="${MAX_PROMPTS:-}"                    # empty = use all prompts

        echo "Using pre-computed embeddings from $EMBD_REPO_DIR/"
        DATA_DIR="data_${TASK}"

        CONVERT_ARGS=(
            --task "$TASK"
            --sft_obj "$SFT_OBJ"
            --gen_model "$REPO_GEN_MODEL"
            --embd_dir "$EMBD_REPO_DIR"
            --output_dir "$DATA_DIR"
        )
        if [[ -n "${MAX_PROMPTS:-}" ]]; then
            CONVERT_ARGS+=(--max_prompts "$MAX_PROMPTS")
        fi

        if ! ls "$DATA_DIR"/EMBD-TRAIN-split_*.npy &>/dev/null; then
            section "Converting Repo Embeddings"
            uv run python3 convert_embeddings.py "${CONVERT_ARGS[@]}"
        else
            echo "Converted data already exists in $DATA_DIR/, skipping conversion."
        fi
    else
        echo "Using pre-computed embeddings from $DATA_DIR/"

        # Generate demo data if no embeddings found
        if ! ls "$DATA_DIR"/EMBD-TRAIN-split_*.npy &>/dev/null; then
            echo "No embedding data found. Generating synthetic demo data..."
            uv run python3 generate_demo_data.py
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Steps 5-8: CPU-friendly reproduction
# ---------------------------------------------------------------------------

section "Step 5: Train Reward Models"
uv run python3 step5_train_rms.py \
    --embed_model_name "$MODEL_NAME" \
    --task "$TASK" \
    --sft_obj "$SFT_OBJ" \
    --gen_pref_model_name "$GEN_PREF_MODEL" \
    --rm_objective "$RM_OBJECTIVE" \
    --consider_first_n "$CONSIDER_FIRST_N" \
    --n_sample "$N_SAMPLE" \
    --training_epochs "$TRAINING_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --seed "$SEED" \
    --annotation_quality "$ANNOTATION_QUALITY" \
    --ensemble_number "$ENSEMBLE_NUMBER" \
    --replacement "$REPLACEMENT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

# Build the checkpoint name (must match step5 naming)
REPLACEMENT_BOOL="False"
TEACHER_CKPT="${OUTPUT_DIR}/XPrompt_mlp_${RM_OBJECTIVE}_$((ENSEMBLE_NUMBER - 1))_seed${SEED}_firstn_${CONSIDER_FIRST_N}_n_${N_SAMPLE}_pair_1_replacement_${REPLACEMENT_BOOL}_epoch${TRAINING_EPOCHS}_lr${LEARNING_RATE}.ckpt"

echo "Teacher checkpoint: $TEACHER_CKPT"
if [[ ! -f "$TEACHER_CKPT" ]]; then
    echo "ERROR: Teacher checkpoint not found at $TEACHER_CKPT"
    echo "Available checkpoints:"
    ls -1 "$OUTPUT_DIR"/*.ckpt 2>/dev/null || echo "  (none)"
    exit 1
fi

section "Step 6: Evaluate Reward Models"
uv run python3 step6_eval_rms.py \
    --embed_model_name "$MODEL_NAME" \
    --task "$TASK" \
    --sft_obj "$SFT_OBJ" \
    --gen_pref_model_name "$GEN_PREF_MODEL" \
    --rm_objective "$RM_OBJECTIVE" \
    --consider_first_n "$CONSIDER_FIRST_N" \
    --n_sample "$N_SAMPLE" \
    --training_epochs "$TRAINING_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --seed "$SEED" \
    --annotation_quality "$ANNOTATION_QUALITY" \
    --ensemble_number "$ENSEMBLE_NUMBER" \
    --replacement "$REPLACEMENT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

section "Step 7: Isotonic Distillation"
STEP7_EXTRA_ARGS=(--student_loss "$STUDENT_LOSS")
if [[ "$STUDENT_LOSS" == "hybrid" ]]; then
    STEP7_EXTRA_ARGS+=(--ranking_lambda "$RANKING_LAMBDA")
    echo "  (hybrid loss: MSE on golden + BT ranking, lambda=$RANKING_LAMBDA)"
fi
if [[ "$NORMALIZE_REWARDS" == "1" ]]; then
    STEP7_EXTRA_ARGS+=(--normalize_rewards)
    echo "  (reward normalization to [0,1] enabled)"
fi
if [[ "$BINARIZE_REWARDS" == "1" ]]; then
    STEP7_EXTRA_ARGS+=(--binarize_rewards)
    echo "  (binary rewards: golden > median -> 1, else -> 0)"
fi
uv run python3 step7_isotonic_distillation.py \
    --embed_model_name "$MODEL_NAME" \
    --task "$TASK" \
    --sft_obj "$SFT_OBJ" \
    --gen_pref_model_name "$MODEL_NAME" \
    --rm_objective "$RM_OBJECTIVE" \
    --teacher_ckpt "$TEACHER_CKPT" \
    --distill_epochs "$DISTILL_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "${STEP7_EXTRA_ARGS[@]}"

STUDENT_CKPT="${OUTPUT_DIR}/distilled_rm_${TASK}_${RM_OBJECTIVE}.ckpt"
ISOTONIC_MODEL="${OUTPUT_DIR}/isotonic_map.joblib"

section "Step 8: Validate Distillation"
STEP8_EXTRA_ARGS=()
if [[ "$NORMALIZE_REWARDS" == "1" ]]; then
    STEP8_EXTRA_ARGS+=(--normalize_rewards)
fi
if [[ "$BINARIZE_REWARDS" == "1" ]]; then
    STEP8_EXTRA_ARGS+=(--binarize_rewards)
fi
uv run python3 step8_validate_distillation.py \
    --embed_model_name "$MODEL_NAME" \
    --task "$TASK" \
    --sft_obj "$SFT_OBJ" \
    --gen_pref_model_name "$MODEL_NAME" \
    --teacher_ckpt "$TEACHER_CKPT" \
    --student_ckpt "$STUDENT_CKPT" \
    --isotonic_model "$ISOTONIC_MODEL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "${STEP8_EXTRA_ARGS[@]}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section "Pipeline Complete"
echo "Output directory: $OUTPUT_DIR/"
echo ""
echo "Key artifacts:"
ls -lh "$OUTPUT_DIR"/*.ckpt "$OUTPUT_DIR"/*.joblib "$OUTPUT_DIR"/*.json 2>/dev/null || true
echo ""
echo "Done!"
