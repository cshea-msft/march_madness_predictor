#!/usr/bin/env python
import os
import argparse
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a shell command and log results."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        process = subprocess.run(
            command, 
            shell=True, 
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Command completed successfully.")
        logger.info(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the complete March Madness training workflow")
    parser.add_argument("--skip-data-collection", action="store_true", 
                      help="Skip historical data collection step")
    parser.add_argument("--skip-fine-tuning", action="store_true",
                      help="Skip model fine-tuning step")
    parser.add_argument("--epochs", type=int, default=4,
                      help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Training batch size")
    parser.add_argument("--data-file", type=str, default="historical_march_madness.csv",
                      help="Path to historical data file")
    parser.add_argument("--model-path", type=str, default="fine_tuned_model",
                      help="Path to save fine-tuned model")
    parser.add_argument("--no-fine-tune", action="store_true",
                      help="Run predictions with base model instead of fine-tuned model")
    args = parser.parse_args()
    
    # Step 1: Collect historical data
    if not args.skip_data_collection:
        logger.info("Step 1: Collecting historical NCAA Tournament data")
        
        if not run_command(
            f"python collect_historical_data.py --output {args.data_file}",
            "Historical data collection"
        ):
            logger.error("Historical data collection failed. Exiting.")
            return
    else:
        logger.info("Skipping historical data collection step")
    
    # Step 2: Fine-tune the model
    if not args.skip_fine_tuning:
        logger.info("Step 2: Fine-tuning the BERT model on historical data")
        
        if not run_command(
            f"python finetune_model.py --data_file {args.data_file} --epochs {args.epochs} "
            f"--batch_size {args.batch_size} --model_save_path {args.model_path}",
            "Model fine-tuning"
        ):
            logger.error("Model fine-tuning failed. Exiting.")
            return
    else:
        logger.info("Skipping model fine-tuning step")
    
    # Step 3: Run predictions with the fine-tuned model
    logger.info("Step 3: Running predictions with the model")
    
    # Determine whether to use fine-tuned model or not
    fine_tune_flag = "" if args.no_fine_tune else "--fine-tuned-model"
    
    if not run_command(
        f"python run_predictions.py {fine_tune_flag}",
        "Running predictions"
    ):
        logger.error("Predictions failed.")
        return
        
    logger.info("Complete workflow finished successfully!")

if __name__ == "__main__":
    main() 