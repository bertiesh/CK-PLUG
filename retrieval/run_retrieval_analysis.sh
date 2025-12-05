#!/bin/bash
# run_retrieval_analysis.sh
# Script to run retrieval quality analysis

# Set paths
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_PATH="./data/nq_dev.jsonl"
OUTPUT_DIR="./results/retrieval_quality_analysis"

# Run analysis
python retrieval_quality_analysis.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_examples 1000 \
    --n_quintiles 5 \
    --alpha 0.5 \
    --methods baseline ck_plug conf_filter

echo "Analysis complete! Results saved to $OUTPUT_DIR"