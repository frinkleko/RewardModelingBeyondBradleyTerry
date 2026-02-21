#!/bin/bash
# Master script for Isotonic Distillation of ORMs

# Update these paths to match your actual checkpoint from Step 5
TEACHER_CKPT="temp_out_dir/XPrompt_mlp_bt_0_seed6_firstn_2_n_10000_pair_1_replacement_replacement_false_epoch30_lr0.001.ckpt"
TASK="helpful"
OUTPUT_DIR="distill_out"

echo "-------------------------------------------------------------------"
echo "Phase 1: Fitting Isotonic Regression & Distilling into Student MLP"
echo "-------------------------------------------------------------------"
# This generates:
# 1. isotonic_map.joblib (The mapping function)
# 2. distilled_rm_helpful_bt.ckpt (The standalone student model)
python3 step7_isotonic_distillation.py 
    --teacher_ckpt "$TEACHER_CKPT" 
    --task "$TASK" 
    --output_dir "$OUTPUT_DIR" 
    --distill_epochs 20 
    --n_samples_for_fit 10000

echo ""
echo "-------------------------------------------------------------------"
echo "Phase 2: Validating Both Scalar Reward Models"
echo "-------------------------------------------------------------------"
# This compares:
# - Model A: Teacher ORM + Isotonic Map (Calibration)
# - Model B: Student MLP (Distillation)
python3 step8_validate_distillation.py 
    --teacher_ckpt "$TEACHER_CKPT" 
    --student_ckpt "$OUTPUT_DIR/distilled_rm_${TASK}_bt.ckpt" 
    --isotonic_model "$OUTPUT_DIR/isotonic_map.joblib" 
    --task "$TASK"
