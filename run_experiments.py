import json
import time
import config as cfg
import LungSegmentation2D_Unet_BBResnet.Main_Resnet as Main_Resnet  # This module now only provides training_job() without auto-running it
import os

def load_param_options(json_path="param_options.json"):
    with open(json_path, "r") as f:
        return json.load(f)

def update_config(params):
    # Update each config variable from the parameter dictionary.
    for key, value in params.items():
        setattr(cfg, key, value)

def run_experiments():
    PARAM_OPTIONS = load_param_options()
    print(f"Loaded {len(PARAM_OPTIONS)} parameter sets.")
    
    # Infinite loop: once you finish the list, start over
    while True:
        for i, params in enumerate(PARAM_OPTIONS, start=1):
            update_config(params)
            print("\n====================================")
            print(f"Starting run {i} with the following config:")
            print(f"  DATA_PATH:         {cfg.DATA_PATH}")
            print(f"  IMAGE_SIZE:        {cfg.IMAGE_SIZE}")
            print(f"  IMG_AUGMENTATION:  {cfg.IMG_AUGMENTATION}")
            print(f"  MASK_AUGMENTATION: {cfg.MASK_AUGMENTATION}")
            print(f"  BATCH_SIZE:        {cfg.BATCH_SIZE}")
            print(f"  NUM_EPOCHS:        {cfg.NUM_EPOCHS}")
            print(f"  TRAIN_TEST_SPLIT:  {cfg.TRAIN_TEST_SPLIT}")
            print(f"  LEARNING_RATE:     {cfg.LEARNING_RATE}")
            print(f"  BACKBONE:          {cfg.BACKBONE}")
            print(f"  MODEL_SAVE_PATH_TEMPLATE: {cfg.MODEL_SAVE_PATH_TEMPLATE}")
            print(f"  MATLAB_SAVE_X_TRAIN: {cfg.MATLAB_SAVE_X_TRAIN}")
            print(f"  MATLAB_SAVE_X_VAL:   {cfg.MATLAB_SAVE_X_VAL}")
            print(f"  MATLAB_SAVE_Y_TRAIN: {cfg.MATLAB_SAVE_Y_TRAIN}")
            print(f"  MATLAB_SAVE_Y_VAL:   {cfg.MATLAB_SAVE_Y_VAL}")
            print("====================================\n")
            
            try:
                # Run the training job for the current configuration.
                Main_Resnet.training_job()
            except KeyboardInterrupt:
                print("Training job was interrupted by user. Exiting experiment queue.")
                return
            
            # Optional: Sleep between runs to allow for manual intervention or system cool-down.
            time.sleep(5)  # Adjust the sleep duration as needed.

if __name__ == "__main__":
    run_experiments()