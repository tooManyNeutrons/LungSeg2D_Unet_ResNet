# config.py
import os
import numpy as np

# Create a folder named 'LungSegmentation2D_Unet_BBResnet_Outputs' in the parent directory of this file
OUTPUT_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'LungSegmentation2D_Unet_BBResnet_Outputs')

# ====================================================
# DATA LOADING & PREPROCESSING PARAMETERS
# ====================================================
# Path to the directory with your data and image size settings.
DATA_PATH = 'Neonatal Test Data - August 1 2024'
IMAGE_SIZE = 256  # assumed square images

# ====================================================
# DATA AUGMENTATION PARAMETERS
# ====================================================
# Parameters for image augmentation (applied to the input images)
IMG_AUGMENTATION = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.05,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'vertical_flip': False,
    'fill_mode': 'nearest'
}

# Parameters for mask augmentation.
# Note: The preprocessing_function ensures masks are binary and converted to float32.
MASK_AUGMENTATION = {
    'rotation_range': 15,
    'width_shift_range': 0.05,
    'height_shift_range': 0.05,
    'shear_range': 0.05,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'fill_mode': 'constant',
    'cval': 0,
    # Do we need to preprocess the masks? If we do, then uncomment 1st line and if we don't comment 2nd line.
    # 'preprocessing_function': lambda x: np.where(x > 0.1, 1, 0).astype('float32')
    'preprocessing_function': None
}

# ====================================================
# TRAINING PARAMETERS
# ====================================================
BATCH_SIZE = 16
NUM_EPOCHS = 5
TRAIN_TEST_SPLIT = 0.25
LEARNING_RATE = 1e-4
BACKBONE = 'resnet50'

# ====================================================
# FILE SAVING & OUTPUT PARAMETERS
# ====================================================
MODEL_SAVE_PATH_TEMPLATE = 'Resnet_model_Xe_2D_Vent_{}epochs.hdf5'
MATLAB_SAVE_X_TRAIN = 'Xe_X_train.mat'
MATLAB_SAVE_X_VAL = 'Xe_X_val.mat'
MATLAB_SAVE_Y_TRAIN = 'Xe_Y_train.mat'
MATLAB_SAVE_Y_VAL = 'Xe_Y_val.mat'
