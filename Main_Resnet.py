import os
os.environ["SM_FRAMEWORK"] = "tf.keras"  # Must be set before any other imports
import random
import time
import datetime

# Third‑Party Libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import nibabel as nib
from scipy.io import savemat
from sklearn.model_selection import train_test_split
import seaborn as sns

# If you want Hausdorff distance, install medpy:
try:
    from medpy.metric.binary import hd
    USE_MEDPY = True
except ImportError:
    print("medpy not installed; Hausdorff distance will be skipped.")
    USE_MEDPY = False

# TensorFlow / Keras Imports
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils as keras_utils
if not hasattr(keras_utils, "generic_utils"):
    keras_utils.generic_utils = type("dummy", (), {"get_custom_objects": keras_utils.get_custom_objects})

# Segmentation Models
import segmentation_models as sm

# Local Modules
import HelpFunctions as HF  # Contains NiftiSequence and helper functions for saving masks/plots
from matplotlib.widgets import Slider  # For interactive visualization

# Import configuration
import config as cfg

# ====================================================
# TRAINING JOB FUNCTION
# ====================================================
def training_job():
    print("Starting training job...")

    # Create a unique output directory based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.OUTPUT_BASE_DIR, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------- DATA GENERATION --------------------
    # Build list of subject directories from DATA_PATH
    subject_dirs = [os.path.join(cfg.DATA_PATH, d) for d in os.listdir(cfg.DATA_PATH)
                    if os.path.isdir(os.path.join(cfg.DATA_PATH, d))]
    print("Total subjects found:", len(subject_dirs))
    
    # Split subject directories into training and validation sets
    train_dirs, val_dirs = train_test_split(subject_dirs, test_size=cfg.TRAIN_TEST_SPLIT, random_state=42)
    print("Train subjects:", len(train_dirs))
    print("Validation subjects:", len(val_dirs))
    
    # -------------------- GENERATORS --------------------
    # Note: If each subject has 220–288 slices, a large batch size (e.g. 16) will load thousands of images at once.
    # To mitigate memory issues, consider reducing cfg.BATCH_SIZE in config.py.
    train_generator = HF.NiftiSequence(
        train_dirs,
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMAGE_SIZE,
        img_aug=cfg.IMG_AUGMENTATION,
        mask_aug=cfg.MASK_AUGMENTATION,
        augment=True,    # Enable augmentation for training
        shuffle=True
    )
    
    val_generator = HF.NiftiSequence(
        val_dirs,
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMAGE_SIZE,
        img_aug=cfg.IMG_AUGMENTATION,
        mask_aug=cfg.MASK_AUGMENTATION,
        augment=False,   # No augmentation for validation
        shuffle=False
    )
    
    # Optionally visualize a few augmented samples from the training generator.
    # (Since Sequence objects are not iterators, we sample a random batch via indexing.)
    HF.visualize_augmented_samples_overlay(train_generator, num_samples=5)
    HF.visualize_augmented_samples(train_generator, num_samples=2)
    
    # -------------------- MODEL DEFINITION --------------------
    BACKBONE = cfg.BACKBONE
    preprocess_input = sm.get_preprocessing(BACKBONE)
    model = sm.Unet(
        BACKBONE,
        encoder_weights='imagenet',
        input_shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3),
        classes=1,
        activation='sigmoid'
    )
    
    def dice_loss(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(y_pred_f) + smooth)
    
    def bce_dice_loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        dsc = dice_loss(y_true, y_pred)
        return bce + dsc
    
    model.compile(optimizer=Adam(learning_rate=cfg.LEARNING_RATE),
                  loss=bce_dice_loss,
                  metrics=[sm.metrics.iou_score])
    print(model.summary())
    
    # -------------------- MODEL TRAINING --------------------
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint(os.path.join(output_dir, 'best_model.keras'),
                        monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=cfg.NUM_EPOCHS,
        callbacks=callbacks
    )
    
    # -------------------- SAVING TRAINING PLOTS & METRICS --------------------
    plots_dir = os.path.join(output_dir, "training_plots")
    HF.save_training_plots(history, output_dir=plots_dir)
    
    # -------------------- PREDICTION & MASK SAVING --------------------
    Y_pred_all = model.predict(val_generator, steps=len(val_generator))
    if Y_pred_all.shape[0] > 0:
        # Convert predictions to a volume (squeeze the channel dimension and transpose so that slices are along the 3rd dimension)
        pred_volume = np.transpose(Y_pred_all.squeeze(-1), (1, 2, 0))
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        HF.save_masks(pred_volume,
                      mat_path=os.path.join(output_dir, "final_gen_mask.mat"),
                      nifti_path=os.path.join(output_dir, "final_gen_mask.nii.gz"),
                      png_dir=masks_dir)
    else:
        print("No predictions were made; skipping mask saving.")
    
    # -------------------- EVALUATION & VISUALIZATION ON ONE BATCH --------------------
    batch_idx = random.randint(0, len(val_generator) - 1)
    X_val_batch, Y_val_batch = val_generator[batch_idx]
    Y_pred_batch = model.predict(X_val_batch)
    
    HF.plot_validation_dice(Y_val_batch, Y_pred_batch, output_dir=plots_dir)
    
    Y_pred_thresholded = (Y_pred_batch > 0.5).astype(np.uint8)
    HF.evaluate_and_save_segmentation_plots(
        Y_true=Y_val_batch,
        Y_pred_probs=Y_pred_batch,
        Y_pred_bin=Y_pred_thresholded,
        output_dir=plots_dir,
        prefix="val"
    )
    
    sample_val_idx = random.randint(0, X_val_batch.shape[0] - 1)
    HF.visualize_image_and_mask(
        X_val_batch[sample_val_idx][:, :, 0],
        Y_val_batch[sample_val_idx][:, :, 0],
        title="Ground Truth"
    )
    HF.visualize_image_and_mask(
        X_val_batch[sample_val_idx][:, :, 0],
        Y_pred_thresholded[sample_val_idx][:, :, 0],
        title="Prediction"
    )
    
    # -------------------- SAVE MODEL --------------------
    model_save_path = os.path.join(output_dir, cfg.MODEL_SAVE_PATH_TEMPLATE.format(cfg.NUM_EPOCHS))
    model.save(model_save_path)
    
    print("Training job completed. Outputs saved in:", output_dir)

def main():
    start_time = time.time()
    training_job()
    elapsed = time.time() - start_time
    print(f"Training job finished in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
