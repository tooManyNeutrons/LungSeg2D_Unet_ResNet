# LungSeg2D_Unet_ResNet

This repository provides a 2D U-Net segmentation pipeline (with a ResNet-50 backbone) for neonatal lung MRI data.

---------------------------------------------------------------------
Prerquisites
---------------------------------------------------------------------
- Python 3.8 or later  
- Git  
- A machine with sufficient memory/GPU for training (optional, but recommended)

---------------------------------------------------------------------
Getting Started
---------------------------------------------------------------------

1. Clone this Repository

   In your terminal or command prompt, run:
   
       git clone https://github.com/Dhyey-Patel28/LungSeg2D_Unet_ResNet.git
       cd LungSeg2D_Unet_ResNet

2. Create and Activate a Virtual Environment

   For Windows (PowerShell):

       python -m venv venv
       .\venv\Scripts\activate

   For macOS / Linux (Bash):

       python3 -m venv venv
       source venv/bin/activate

3. Install Requirements

   With the virtual environment active, install the dependencies:

       pip install -r requirements.txt

4. Set Up Your Data Path

   Open the file `config.py` and modify the following line so that it points to your data folder (where the files such as `proton.nii` and `mask.nii` live):

       DATA_PATH = 'Neonatal Test Data - August 1 2024'

   Replace "Neonatal Test Data - August 1 2024" with the path to your own data directory.

---------------------------------------------------------------------
Single Training Run
---------------------------------------------------------------------

To run a single training job (using the settings in `config.py`), execute:

       python Main_Resnet_Xe.py

This will start training based on your current configuration.

---------------------------------------------------------------------
Running Experiments Indefinitely with Custom Parameters
---------------------------------------------------------------------

In addition to a single training job, you can run experiments indefinitely with different configurations by using the provided `run_experiments.py` script along with the `param_options.json` file.

1. Modify Parameter Options

   The file `param_options.json` contains a list of parameter sets (in JSON format). Each parameter set may include values for:
   - Data Path and Image Size (e.g., `DATA_PATH` and `IMAGE_SIZE`)
   - Data Augmentation Settings (for `IMG_AUGMENTATION` and `MASK_AUGMENTATION`)
   - Training Parameters (such as `BATCH_SIZE`, `NUM_EPOCHS`, `TRAIN_TEST_SPLIT`, and `LEARNING_RATE`)
   - Model Backbone (e.g., `"resnet50"`)
   - File Saving Parameters (e.g., MATLAB save file names) which remain constant across runs

   For example, a parameter set might look like this:

       [
         {
           "DATA_PATH": "Neonatal Test Data - August 1 2024",
           "IMAGE_SIZE": 256,
           "IMG_AUGMENTATION": {
             "rotation_range": 15,
             "width_shift_range": 0.1,
             "height_shift_range": 0.1,
             "shear_range": 0.05,
             "zoom_range": 0.1,
             "horizontal_flip": true,
             "vertical_flip": false,
             "fill_mode": "nearest"
           },
           "MASK_AUGMENTATION": {
             "rotation_range": 15,
             "width_shift_range": 0.05,
             "height_shift_range": 0.05,
             "shear_range": 0.05,
             "zoom_range": 0.1,
             "horizontal_flip": true,
             "fill_mode": "constant",
             "cval": 0,
             "preprocessing_function": null
           },
           "BATCH_SIZE": 8,
           "NUM_EPOCHS": 3,
           "TRAIN_TEST_SPLIT": 0.25,
           "LEARNING_RATE": 0.0001,
           "BACKBONE": "resnet50",
           "MODEL_SAVE_PATH_TEMPLATE": "Resnet_model_Xe_2D_Vent_{}epochs.hdf5",
           "MATLAB_SAVE_X_TRAIN": "Xe_X_train.mat",
           "MATLAB_SAVE_X_VAL": "Xe_X_val.mat",
           "MATLAB_SAVE_Y_TRAIN": "Xe_Y_train.mat",
           "MATLAB_SAVE_Y_VAL": "Xe_Y_val.mat"
         }
       ]

   Feel free to add or modify parameter sets to suit your experimental needs.

2. Run the Experiment Queue

   Use the `run_experiments.py` script to run the experiments indefinitely. This script:
   - Loads parameter sets from `param_options.json`
   - Iterates over each configuration and updates `config.py` accordingly
   - Calls `Main_Resnet_Xe.training_job()` for a single run
   - Optionally waits (for example, 5 seconds) between runs
   - Repeats the process indefinitely

   To start the experiment queue, run:

       python run_experiments.py

   The script will print out the configuration for each run and execute the training job. Once all configurations have been used, the process will start over.

---------------------------------------------------------------------
Stopping The Model
---------------------------------------------------------------------

To interrupt training or the experiment queue, press `Ctrl + C` in your terminal or command prompt.

---------------------------------------------------------------------
Repository Structure
---------------------------------------------------------------------

- **Main_Resnet_Xe.py**  
  The main script that sets up the data generators, model, and executes a single training job.
- **HelpFunctions.py**  
  Utility functions for data loading, plotting, and evaluation metrics.
- **config.py**  
  Configuration script with hyperparameters, data paths, and augmentation settings.
- **param_options.json**  
  JSON file containing a list of parameter sets for running experiments with different configurations.
- **run_experiments.py**  
  Script that loads parameter sets from `param_options.json` and repeatedly runs training jobs indefinitely.
- **requirements.txt**  
  Python libraries needed to run the segmentation pipeline.
- **UnitTest_Model.py**  
  Example test script or placeholder for unit tests.
- **Test_Model.py**  
  Additional script for testing or example usage.

---------------------------------------------------------------------
Contributing
---------------------------------------------------------------------

Feel free to open an issue or submit a pull request if you have improvements or find bugs.