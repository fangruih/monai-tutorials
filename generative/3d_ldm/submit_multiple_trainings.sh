#!/bin/bash

# Define the weight combinations to try
AGE_LOSS_WEIGHTS=(0.0001 0.001 0.01)
CYCLE_LOSS_WEIGHTS=(0.1 1)
TRANSFER_LOSS_WEIGHTS=(0.1 1)

# Base config file path
BASE_CONFIG="config/cycle/generated/config_cycle_4.json"

# Directory to store generated configs
CONFIG_DIR="config/cycle/generated/experiments"
mkdir -p $CONFIG_DIR

# Python script to modify JSON
cat << EOF > modify_json.py
import json
import sys

def modify_json(input_file, output_file, age_weight, cycle_weight, transfer_weight):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    data['cycle_param']['age_loss_weight'] = float(age_weight)
    data['cycle_param']['cycle_loss_weight'] = float(cycle_weight)
    data['cycle_param']['transfer_loss_weight'] = float(transfer_weight)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    modify_json(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
EOF

# Loop through all combinations, create configs, and submit jobs
for age_weight in "${AGE_LOSS_WEIGHTS[@]}"; do
    for cycle_weight in "${CYCLE_LOSS_WEIGHTS[@]}"; do
        for transfer_weight in "${TRANSFER_LOSS_WEIGHTS[@]}"; do
            # Create a new config file with modified weights
            NEW_CONFIG="$CONFIG_DIR/config_cycle_4_age${age_weight}_cycle${cycle_weight}_transfer${transfer_weight}.json"
            python modify_json.py "$BASE_CONFIG" "$NEW_CONFIG" "$age_weight" "$cycle_weight" "$transfer_weight"

            echo "Created config file: $NEW_CONFIG"
            echo "Submitting job with weights: AGE=$age_weight, CYCLE=$cycle_weight, TRANSFER=$transfer_weight"
            sbatch submit_training.sh $NEW_CONFIG
        done
    done
done

echo "All jobs submitted."

# Clean up the temporary Python script
rm modify_json.py
