# test_main_resnet.py

import time

# Import your config to override certain parameters
import config as cfg

# Import your main script, which has training_job()
import Main_Resnet as Main_Resnet  # Ensure the 'while True:' loop is removed or guarded in Main_Resnet_Xe

def run_test():
    """
    This function calls `training_job()` exactly once with minimal epochs
    to test if everything works end-to-end.
    """
    # Optionally override some config parameters for a quicker run:
    cfg.NUM_EPOCHS = 1       # Only 1 epoch for testing
    cfg.BATCH_SIZE = 4       # Smaller batch size
    # You can also point DATA_PATH to a small test dataset.
    # cfg.DATA_PATH = "path/to/a_small_test_folder"

    print("=== Starting a test run of training_job() ===")
    start_time = time.time()

    # Now just call the training_job() function directly.
    Main_Resnet.training_job()

    elapsed = time.time() - start_time
    print(f"=== Test run finished in {elapsed:.2f} seconds. ===")

if __name__ == "__main__":
    run_test()