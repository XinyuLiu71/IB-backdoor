#!/bin/bash

# Default parameters
DATASET="cifar10"
ATTACK_TYPE="badnet"
POISON_PERCENTAGE="0.1"
TARGET_CLASS="0"
MODEL="resnet18"
NOISE_STD_XT="0.4"
NOISE_STD_TY="0.4"
OUTPUT_DIR="results/${DATASET}/${ATTACK_TYPE}/ob_infoNCE_$(date +%m_%d)_${POISON_PERCENTAGE}_${NOISE_STD_XT}+${NOISE_STD_TY}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --attack_type)
      ATTACK_TYPE="$2"
      shift 2
      ;;
    --poison_percentage)
      POISON_PERCENTAGE="$2"
      shift 2
      ;;
    --target_class)
      TARGET_CLASS="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --noise_std_xt)
      NOISE_STD_XT="$2"
      shift 2
      ;;
    --noise_std_ty)
      NOISE_STD_TY="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Generate backdoor dataset
echo "Generating backdoor dataset..."
cd poison_data_generator
python poison_data_generator.py \
    --attack_type="$ATTACK_TYPE" \
    --dataset="$DATASET" \
    --poison_percentage="$POISON_PERCENTAGE" \
    --target_class="$TARGET_CLASS"
cd ..

# Step 2: Create FFCV dataloader
echo "Creating FFCV dataloader..."
python ffcv_writer4.py \
    --output_path="data/${DATASET}/${ATTACK_TYPE}/${POISON_PERCENTAGE}" \
    --dataset=all \
    --train_data_path="data/${DATASET}/${ATTACK_TYPE}/${POISON_PERCENTAGE}/${ATTACK_TYPE}_${POISON_PERCENTAGE}.npz" \
    --test_data_path="data/${DATASET}/${ATTACK_TYPE}/${POISON_PERCENTAGE}/test_data.npz"

# Step 3: Train model and observe MI
echo "Training model and observing MI..."
python ffcv_observeMI.py \
    --outputs_dir="$OUTPUT_DIR" \
    --train_data_path="data/${DATASET}/${ATTACK_TYPE}/${POISON_PERCENTAGE}/train_data.beton" \
    --test_data_path="data/${DATASET}/${ATTACK_TYPE}/${POISON_PERCENTAGE}/test_data.beton" \
    --sample_data_path="data/${DATASET}/${ATTACK_TYPE}/${POISON_PERCENTAGE}/train_data" \
    --model="$MODEL" \
    --noise_std_xt="$NOISE_STD_XT" \
    --noise_std_ty="$NOISE_STD_TY"

# Step 4: Plot results
echo "Plotting results..."
python plot.py --directory="$OUTPUT_DIR"

echo "Experiment completed! Results saved in $OUTPUT_DIR" 