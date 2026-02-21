#!/bin/bash

# Configuration
TASK="helpful"
MODEL_NAME="gemma2b"
DATASET="hh-rlhf-helpful"
OUTPUT_DIR="distill_out"
DATA_DIR="data"
SEED=0

echo "Starting Calibrated Reward Modeling Pipeline..."

# Step 1: SFT (Skipping if already done, but showing command)
# python step1_sft.py --model_name $MODEL_NAME --dataset $DATASET-gpt4 --output_dir $OUTPUT_DIR

# Step 2: Generate Samples (Skipping if already done, but showing command)
# python step2_gen_sample.py --model_name $MODEL_NAME --dataset $DATASET --eval_dataset $DATASET --output_dir $OUTPUT_DIR

# Step 3: Reward Annotation (Assuming rewards are available or generated)
# python step3_reward_annotation.py ...

# Step 4: Generate Embeddings (Assuming embeddings are generated)
# python step4_gen_embeddings.py --model_name $MODEL_NAME --task $TASK --output_dir $OUTPUT_DIR

# Step 5: Train Reward Model
echo "Step 5: Training Reward Model..."
python step5_train_rms.py 
    --task $TASK 
    --output_dir $OUTPUT_DIR 
    --data_dir $DATA_DIR 
    --ensemble_number 1 
    --training_epochs 5 
    --seed $SEED

# Step 6: Evaluate Base Reward Model
echo "Step 6: Evaluating Base Reward Model..."
python step6_eval_rms.py 
    --task $TASK 
    --output_dir $OUTPUT_DIR 
    --data_dir $DATA_DIR 
    --ensemble_number 1 
    --seed $SEED

# Step 7: Calibrate Reward Model with Isotonic Regression
echo "Step 7: Calibrating Reward Model with Isotonic Regression..."
python step7_calibrate_rm.py 
    --task $TASK 
    --output_dir $OUTPUT_DIR 
    --data_dir $DATA_DIR 
    --ensemble_number 1 
    --seed $SEED

# Step 8: Evaluate Calibrated Reward Model
echo "Step 8: Evaluating Calibrated Reward Model..."
CALIBRATOR_PATH="${OUTPUT_DIR}/isotonic_calibrator_bt_0_seed${SEED}.joblib"
python step6_eval_rms.py 
    --task $TASK 
    --output_dir $OUTPUT_DIR 
    --data_dir $DATA_DIR 
    --ensemble_number 1 
    --seed $SEED 
    --calibrator_path $CALIBRATOR_PATH

echo "Pipeline completed successfully!"
