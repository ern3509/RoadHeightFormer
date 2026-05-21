#!/bin/bash

# RoadHeightFormer Training Script with Config Support
# Usage: ./run_experiments.sh [config_file] [additional_args]

CONFIG_FILE=${1:-"config.yaml"}
shift  # Remove first argument so remaining args are passed to train.py

echo "====================="
echo "Running experiments..."
echo "====================="
echo "Using configuration file: $CONFIG_FILE"
echo "Additional arguments: $@"

# Run training with config file
python train.py --config "$CONFIG_FILE" "$@"

# Example usage:
# ./run_experiments.sh config.yaml --epochs 100 --batch_size 16
# ./run_experiments.sh experiments/experiment_1.yaml
# ./run_experiments.sh config.yaml --name_run "test_run" --notes "Testing new loss"