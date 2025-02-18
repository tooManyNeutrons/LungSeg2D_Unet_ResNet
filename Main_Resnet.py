import os
os.environ["SM_FRAMEWORK"] = "tf.keras" # Must be set before any other imports
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
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
# If you want Hausdorff distance, install medpy:
# pip install medpy
try:
    from medpy.metric.binary import hd
    USE_MEDPY = True
except ImportError:
    print("medpy not installed; Hausdorff distance will be skipped.")
    USE_MEDPY = False

# TensorFlow / Keras Imports
import tensorflow as tf
from keras import backend as K
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.optimizers import Adam
from keras import utils as keras_utils
# Monkey-patch: If 'generic_utils' is missing, map it to get_custom_objects
if not hasattr(keras_utils, "generic_utils"):
    keras_utils.generic_utils = type("dummy", (), {"get_custom_objects": keras_utils.get_custom_objects})

# Segmentation Models
import segmentation_models as sm

# Local Modules
from LungSegmentation2D_Unet_BBResnet import HelpFunctions as HF  # Assumes this module handles data I/O
from matplotlib.widgets import Slider  # For interactive visualization

# ====================================================
# Import configuration
# ====================================================
from LungSegmentation2D_Unet_BBResnet import config as cfg

# ====================================================
# TRAINING JOB FUNCTION
# ====================================================
def training_job():
    print("Starting training job...")

    # Create a unique output directory based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.OUTPUT_BASE_DIR, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # DATA LOADING & PREPROCESSING
    # ----------------------------------------------------------
    proton_images, masks, subject_paths = HF.read_2D_nifti_Xe_H_masks(cfg.DATA_PATH, Im_size=cfg.IMAGE_SIZE)
    
    print("Proton Images Shape:", proton_images.shape)
    print("Masks Shape:", masks.shape)
    print("Subjects Loaded:", len(subject_paths))
    
    print("Mask min:", masks.min(), "Mask max:", masks.max(), "unique:", np.unique(masks))
    
    images = np.expand_dims(proton_images, axis=-1)
    masks  = np.expand_dims(masks, axis=-1)
    # masks = masks[:, :, :, ::-1, :] # Reversing the masks
    masks = np.where(masks > 0.5, 1, 0).astype(np.uint8)
    num_subjects, H, W, num_slices, _ = images.shape
    
    # ----------------------------------------------------------
    # SUBJECT-WISE SPLIT ON INDICES
    # ----------------------------------------------------------
    subject_indices = np.arange(num_subjects)
    train_subj_idx, val_subj_idx = train_test_split(
        subject_indices, 
        test_size=cfg.TRAIN_TEST_SPLIT, 
        random_state=42
    )

    # Step-by-step explanation:
    # 1) We'll replace the old flattening approach (X_train = X_train.reshape(-1, H, W, 1)) 
    #    with a manual building of flattened arrays + storing slice maps.
    # 2) This ensures we can later reconstruct the volumes properly.

    # Create Python lists for flattened training & validation slices:
    X_train_list = []
    Y_train_list = []
    slice_map_train = {}  # subject -> (start_index, end_index) in flattened X_train
    train_offset = 0

    # ========== LOOP OVER TRAIN SUBJECTS ==========
    for s in train_subj_idx:
        start_idx = train_offset
        end_idx   = train_offset + num_slices
        slice_map_train[s] = (start_idx, end_idx)

        # Append each slice from subject s to the lists
        for slc in range(num_slices):
            X_train_list.append(images[s, :, :, slc, :])  # shape (H, W, 1)
            Y_train_list.append(masks[s, :, :, slc, :])
        train_offset += num_slices

    # Convert the Python lists to NumPy arrays => shape (total_train_slices, H, W, 1)
    X_train = np.stack(X_train_list, axis=0)
    Y_train = np.stack(Y_train_list, axis=0)

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    
    # ========== LOOP OVER VAL SUBJECTS ==========
    X_val_list = []
    Y_val_list = []
    slice_map_val = {}
    val_offset = 0

    for s in val_subj_idx:
        start_idx = val_offset
        end_idx   = val_offset + num_slices
        slice_map_val[s] = (start_idx, end_idx)

        for slc in range(num_slices):
            X_val_list.append(images[s, :, :, slc, :])
            Y_val_list.append(masks[s, :, :, slc, :])
        val_offset += num_slices

    # Convert Python lists for val => arrays
    X_val = np.stack(X_val_list, axis=0)
    Y_val = np.stack(Y_val_list, axis=0)

    # Now we have X_train, Y_train, X_val, Y_val in shape => (N_slices, H, W, 1)
    # plus slice_map_train[s] & slice_map_val[s] letting us reassemble slices by subject.

    # ========== CONVERT TO FLOAT & REPLICATE CHANNELS ==========

    # Typically, we scale images to [0..1]
    X_train = X_train.astype('float32') / 255.0
    X_val   = X_val.astype('float32') / 255.0
    Y_train = Y_train.astype('float32')
    Y_val   = Y_val.astype('float32')

    # If ResNet-50, replicate single channel -> 3 channels
    X_train = np.repeat(X_train, 3, axis=-1)  # shape => (N_train_slices, H, W, 3)
    X_val   = np.repeat(X_val,   3, axis=-1)
    
    # ----------------------------------------------------------
    # DATA AUGMENTATION 
    # ----------------------------------------------------------
    img_datagen = ImageDataGenerator(**cfg.IMG_AUGMENTATION)
    mask_datagen = ImageDataGenerator(**cfg.MASK_AUGMENTATION)

    batch_size = cfg.BATCH_SIZE
    seed = 42

    # Flow from Numpy arrays
    img_generator  = img_datagen.flow(X_train,   batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(Y_train,  batch_size=batch_size, seed=seed)

    def train_generator_func(image_gen, mask_gen):
        while True:
            X_batch = next(image_gen)
            Y_batch = next(mask_gen)
            yield (X_batch, Y_batch)

    train_generator = train_generator_func(img_generator, mask_generator)
    
    # Seeing a few augmented samples
    HF.visualize_augmented_samples_overlay(train_generator, num_samples=5)

    # Seeing a few augmented samples
    HF.visualize_augmented_samples(train_generator, num_samples=2)

    # ----------------------------------------------------------
    # BUILD MODEL DEFINITION: 2D U‑NET with RESNET‑50 BACKBONE
    # ----------------------------------------------------------
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
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    def bce_dice_loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        dsc = dice_loss(y_true, y_pred)
        return bce + dsc
    
    model.compile(optimizer=Adam(learning_rate=cfg.LEARNING_RATE),
                  loss=bce_dice_loss,
                  metrics=[sm.metrics.iou_score])
    
    print(model.summary())

    # MODEL TRAINING
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint(os.path.join(output_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True)
    ]
    
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    num_epochs = cfg.NUM_EPOCHS
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, Y_val),
        validation_steps=validation_steps,
        epochs=num_epochs,
        callbacks=callbacks
    )

    # ----------------------------------------------------------
    # SAVE TRAINING PLOTS & METRICS
    # ----------------------------------------------------------
    plots_dir = os.path.join(output_dir, "training_plots")
    HF.save_training_plots(history, output_dir=plots_dir)

    Y_pred = model.predict(X_val)
    HF.plot_validation_dice(Y_val, Y_pred, output_dir=plots_dir)

    # ----------------------------------------------------------
    # EVALUATION & VISUALIZATION
    # ----------------------------------------------------------
    Y_pred_thresholded = (Y_pred > 0.5).astype(np.uint8)
    
    HF.evaluate_and_save_segmentation_plots(
        Y_true=Y_val,
        Y_pred_probs=Y_pred,
        Y_pred_bin=Y_pred_thresholded,
        output_dir=plots_dir,
        prefix="val"
    )
    
    # Visualize 1 random slice
    sample_val_idx = random.randint(0, X_val.shape[0] - 1)
    HF.visualize_image_and_mask(
        X_val[sample_val_idx][:, :, 0],
        Y_val[sample_val_idx][:, :, 0],
        title="Ground Truth"
    )
    HF.visualize_image_and_mask(
        X_val[sample_val_idx][:, :, 0],
        Y_pred_thresholded[sample_val_idx][:, :, 0],
        title="Prediction"
    )
    
    # 3D RECONSTRUCTION NOTE:
    # Below, we do a naive approach: subject i is assumed to be i * num_slices => (i+1)*num_slices
    # But we actually built slice_map_val[s], so you could do:
    #
    #   start_idx, end_idx = slice_map_val[s]
    #   subject_pred_volume = Y_pred_thresholded[start_idx:end_idx]
    #
    # which is more robust if the subject order changed.

    # Do this:
    s = val_subj_idx[0]  # This is the actual subject ID
    start_idx, end_idx = slice_map_val[s]  # Retrieve the stored slice range
    subject_pred_volume = Y_pred_thresholded[start_idx:end_idx]  # shape (num_slices, H, W, 1)

    # Then transpose/reshape so that the slice dimension goes last (H,W,slices):
    subject_pred_volume = np.transpose(subject_pred_volume.squeeze(-1), (1, 2, 0))  
    # shape now is (H, W, num_slices)

    print("subject_pred_volume shape:", subject_pred_volume.shape)  # e.g. (256,256,256)
    
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    HF.save_masks(subject_pred_volume,
                mat_path=os.path.join(output_dir, "final_gen_mask.mat"),
                nifti_path=os.path.join(output_dir, "final_gen_mask.nii.gz"),
                png_dir=masks_dir)
    
    # ----------------------------------------------------------
    # SAVE MODEL AND DATA
    # ----------------------------------------------------------
    model_save_path = os.path.join(output_dir, cfg.MODEL_SAVE_PATH_TEMPLATE.format(num_epochs))
    model.save(model_save_path)

    savemat(os.path.join(output_dir, cfg.MATLAB_SAVE_X_TRAIN), {"X_train": X_train})
    savemat(os.path.join(output_dir, cfg.MATLAB_SAVE_X_VAL),   {"X_val":   X_val})
    savemat(os.path.join(output_dir, cfg.MATLAB_SAVE_Y_TRAIN), {"Y_train": Y_train})
    savemat(os.path.join(output_dir, cfg.MATLAB_SAVE_Y_VAL),   {"Y_val":   Y_val})

    print("Training job completed. Outputs saved in:", output_dir)
    
# ====================================================
# RUN THE TRAINING JOB DAILY (Ensuring No Overlap)
# ====================================================
    
def main():
    
    start_time = time.time()
    training_job()  # Run the training job from start to finish
    elapsed = time.time() - start_time
    # Wait until 24 hours have passed since start
    sleep_time = max(0, 86400 - elapsed)
    print(f"Training job finished in {elapsed:.2f} seconds. Sleeping for {sleep_time:.2f} seconds before next run.")
    time.sleep(sleep_time)
    
    # You can decide whether to run a single training job or a continuous loop.
    # For instance, here we run a single training job.
    start_time = time.time()
    training_job()
    elapsed = time.time() - start_time
    print(f"Training job finished in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()