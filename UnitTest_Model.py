# test_main_resnet.py
import unittest
import time
import config as cfg
import LungSegmentation2D_Unet_BBResnet.Main_Resnet as Main_Resnet  # Must not auto-start the loop

class TestMainResnetXe(unittest.TestCase):
    def test_run_one_epoch(self):
        cfg.NUM_EPOCHS = 1
        cfg.BATCH_SIZE = 4

        start_time = time.time()
        Main_Resnet.training_job()  # run once
        elapsed = time.time() - start_time

        # No time-based checks; just confirm it runs to completion.
        print(f"Test completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    unittest.main()
