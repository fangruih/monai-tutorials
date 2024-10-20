import argparse
import os
import torch
from monai.utils import set_determinism
from pathlib import Path
import json
import pandas as pd
import sys
from datetime import datetime



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluate_fid_3d import calculate_fid_3d
from evaluate_with_counterfactual_translation_multi_age import evaluate_with_counterfactual_translation_multi_age
from evaluate_with_random_condition import evaluate_with_random_condition

from utils import define_instance

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate multiple model checkpoints")
    parser.add_argument(
        "-e",
        "--environment-file",
        type=str,
        required=True,
        help="Path to the environment configuration file"
    )
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        required=True,
        help="Path to the model configuration file"
    )
    parser.add_argument(
        "--checkpoint_paths",
        nargs="+",
        required=True,
        help="List of checkpoint file paths"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate"
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()
    # Load model configuration
    with open(args.config_file, 'r') as f:
        model_config = json.load(f)
    for k, v in model_config.items():
        setattr(args, k, v)
    return args

def evaluate_model_combine(args):
    
    evaluation_common_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    Path(args.evaluation_output_dir).mkdir(parents=True, exist_ok=True)
    excel_filename = f"{evaluation_common_timestamp}.xlsx"
    excel_filepath = os.path.join(args.evaluation_output_dir, excel_filename)


    for checkpoint_path in args.checkpoint_paths:
        
        # LOADING CHECKPOINT
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            continue
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Load the model weights from the checkpoint
            
            unet_model_state_dict = checkpoint['model_state_dict']
            
            # Record additional information in evaluation_results
            evaluation_results = {
                "checkpoint": checkpoint_path,
                "accuracy": 0.0,
                "loss": 0.0,
                "learning_rate": checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] if 'optimizer_state_dict' in checkpoint else None,
                "epoch": checkpoint.get('epoch', None),
                "timestamp": checkpoint.get('timestamp', None),
                "loss_weight": checkpoint.get('loss_weight', None),
                "train_transfer_loss_x_iter": checkpoint.get('train_transfer_loss_x_iter', None),
                "train_transfer_loss_y_iter": checkpoint.get('train_transfer_loss_y_iter', None),
                "train_age_loss_iter": checkpoint.get('train_age_loss_iter', None),
                "train_cycle_loss_iter": checkpoint.get('train_cycle_loss_iter', None),
                "train_cycle_transfer_loss_iter": checkpoint.get('train_cycle_transfer_loss_iter', None),
                "train_weight_loss_iter": checkpoint.get('train_weight_loss_iter', None),
                "train_total_loss_iter": checkpoint.get('train_total_loss_iter', None),
                "step": checkpoint.get('step', None),
            }
        else:
            # If the checkpoint only contains the model state dict, load it directly
            unet_model_state_dict = checkpoint
            
            # Initialize evaluation_results with default values
            evaluation_results = {
                "checkpoint": checkpoint_path,
                
            }
        
        # GENERATING IMAGES, from random and counterfactual conditions
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        os.chdir(parent_dir)
        print(f"Changed working directory to: {os.getcwd()}")
        
        # 1. from counterfactual translation
        converted_images_file_dir = evaluate_with_counterfactual_translation_multi_age(
            age_or_sex="age",
            diffusion_evaluation_checkpoint=unet_model_state_dict,
            args=args,
            age_list=[10, 40, 80],
            common_timestamp=evaluation_common_timestamp
        )
        
        print("converted_images_file_dir", converted_images_file_dir)
        
        # 2. from random condition
        generated_images_file_dir = evaluate_with_random_condition(
            args = args, 
            diffusion_evaluation_checkpoint = unet_model_state_dict,
            common_timestamp=evaluation_common_timestamp
        )
        print("generated_images_file_dir", generated_images_file_dir)
        
        # CALCULATING METRICS
        # 1. Calculate FIDs
        fid_3d_generated = calculate_fid_3d(generated_images_file_dir)
        fid_3d_converted = calculate_fid_3d(converted_images_file_dir)
        
        # 2. Calculate Group Similarity
        group_similarity = calculate_group_similarity(args.data_dir)
        
        
        evaluation_results.update({
            "checkpoint": checkpoint_path,
            "accuracy": 0.0,
            "loss": 0.0,
            "learning_rate": checkpoint['optimizer_state_dict']['param_groups'][0]['lr'],
            "epoch": checkpoint['epoch'],
            "timestamp": checkpoint['timestamp'],
            "loss_weight": checkpoint['loss_weight'],
            "file_path_to_generated_image": "TODO",
            "group_similarity": group_similarity,
            "fid_3d_generated": fid_3d_generated,
            "fid_3d_converted": fid_3d_converted,
        })

        # Save evaluation results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_filename = f"eval_results_{Path(checkpoint_path).stem}_time{timestamp}.xlsx"
        result_filepath = os.path.join(args.output_dir, result_filename)
        
        if os.path.exists(result_filepath):
            # If file exists, read existing data and append new row
            existing_df = pd.read_excel(result_filepath)
            new_df = pd.DataFrame([evaluation_results])
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_excel(result_filepath, index=False)
        else:
            # If file doesn't exist, create new file with the data
            df = pd.DataFrame([evaluation_results])
            df.to_excel(result_filepath, index=False)

    print(f"Evaluation complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_model_combine(args)
