# batch_runs.py
import time
import config as cfg
import LungSegmentation2D_Unet_BBResnet.Main_Resnet as Main_Resnet  # Make sure it does NOT auto-run training on import
import os

# Define different parameter sets you want to try:
# e.g., varying batch size, LR, or # of epochs
PARAM_SETS = [
    {"BATCH_SIZE": 8,  "LEARNING_RATE": 1e-4, "NUM_EPOCHS": 3},
    {"BATCH_SIZE": 16, "LEARNING_RATE": 1e-4, "NUM_EPOCHS": 5},
    {"BATCH_SIZE": 16, "LEARNING_RATE": 5e-5, "NUM_EPOCHS": 8},
    # ... add as many as you like ...
]

def run_queued_experiments():
    # Optionally ask user if they want to proceed
    print(f"About to run {len(PARAM_SETS)} configurations.")
    input("Press Enter to start...")

    for i, params in enumerate(PARAM_SETS, start=1):
        # Update config
        cfg.BATCH_SIZE    = params["BATCH_SIZE"]
        cfg.LEARNING_RATE = params["LEARNING_RATE"]
        cfg.NUM_EPOCHS    = params["NUM_EPOCHS"]

        print(f"\n=== Starting run {i}/{len(PARAM_SETS)} ===")
        print(f"Using BATCH_SIZE={cfg.BATCH_SIZE}, LEARNING_RATE={cfg.LEARNING_RATE}, EPOCHS={cfg.NUM_EPOCHS}")

        # Run your training job exactly once
        start_time = time.time()
        Main_Resnet.training_job()
        elapsed = time.time() - start_time
        print(f"Run {i} completed in {elapsed:.2f} seconds.")

        # (Optional) Wait or ask if user wants to proceed
        user_input = input("Press Enter to proceed to next run, or type 'q' to quit: ")
        if user_input.lower() == 'q':
            print("Exiting early by user request.")
            break

if __name__ == "__main__":
    run_queued_experiments()
