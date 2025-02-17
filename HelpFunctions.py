# HelpFunctions.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import nibabel as nib
from scipy.io import savemat
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

try:
    from medpy.metric.binary import hd
    USE_MEDPY = True
except ImportError:
    print("medpy not installed; Hausdorff distance will be skipped.")
    USE_MEDPY = False

###############################################
# READ 2D NIfTI VOLUMES FOR PROTON & MASK DATA
###############################################
def read_2D_nifti_Xe_H_masks(directory, Im_size=256):
    """
    Reads proton images and corresponding masks from subdirectories in `directory`.
    Automatically selects a mask file if one candidate has a basename exactly "mask.nii".
    
    Returns:
        proton_images: shape (num_subjects, H, W, num_slices)
        masks:         shape (num_subjects, H, W, num_slices)
        subject_paths: list of subfolder paths
    """
    # Gather subdirectories
    subject_dirs = sorted(
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    )

    proton_images_list = []
    masks_list = []
    subject_paths = []

    for subject_dir in subject_dirs:
        # Find files matching proton or mask patterns
        proton_candidates = glob.glob(os.path.join(subject_dir, '*[Pp]roton*.*nii*'))
        mask_candidates   = glob.glob(os.path.join(subject_dir, '*[Mm]ask*.*nii*'))

        # Skip subject if no candidates found
        if not proton_candidates:
            print(f"No proton files found in {subject_dir}. Skipping.")
            continue
        if not mask_candidates:
            print(f"No mask files found in {subject_dir}. Skipping.")
            continue

        # Automatically select the mask file if one candidate has basename "mask.nii"
        selected_mask = None
        for candidate in mask_candidates:
            if os.path.basename(candidate).lower() == "mask.nii":
                selected_mask = candidate
                print(f"\nSubject: {subject_dir}")
                print(f"Automatically selected mask file: {selected_mask}")
                break

        # If no automatic selection, prompt the user as before
        if selected_mask is None:
            if len(mask_candidates) == 1:
                selected_mask = mask_candidates[0]
                print(f"\nSubject: {subject_dir}")
                print(f"Found a single mask candidate: {selected_mask}")
            else:
                print(f"\nSubject: {subject_dir}")
                print("Multiple mask candidates found:")
                for i, mfile in enumerate(mask_candidates):
                    print(f"  {i+1}) {mfile}")
                print("  S) Skip this subject")
                while True:
                    choice = input("Select the mask file by number, or 'S' to skip: ")
                    if choice.lower() == 's':
                        print("Skipping this subject...")
                        selected_mask = None
                        break
                    try:
                        idx = int(choice) - 1
                        if idx < 0 or idx >= len(mask_candidates):
                            print("Invalid choice. Try again.")
                        else:
                            selected_mask = mask_candidates[idx]
                            print(f"Selected mask file: {selected_mask}")
                            break
                    except ValueError:
                        print("Invalid input. Please enter a number or 'S'.")
                if selected_mask is None:
                    continue  # Skip this subject if user chose to skip

        # Handle proton candidates similarly (here we assume one candidate or prompt user)
        if len(proton_candidates) == 1:
            proton_file = proton_candidates[0]
            print(f"Found a single proton candidate: {proton_file}")
        else:
            print(f"\nSubject: {subject_dir}")
            print("Multiple proton candidates found:")
            for i, pfile in enumerate(proton_candidates):
                print(f"  {i+1}) {pfile}")
            print("  S) Skip this subject")
            while True:
                choice = input("Select the proton file by number, or 'S' to skip: ")
                if choice.lower() == 's':
                    print("Skipping this subject...")
                    proton_file = None
                    break
                try:
                    idx = int(choice) - 1
                    if idx < 0 or idx >= len(proton_candidates):
                        print("Invalid choice. Try again.")
                    else:
                        proton_file = proton_candidates[idx]
                        print(f"Selected proton file: {proton_file}")
                        break
                except ValueError:
                    print("Invalid input. Please enter a number or 'S'.")
            if proton_file is None:
                continue

        print(f"\nLoading {proton_file} and {selected_mask}...")
        try:
            proton_data = nib.load(proton_file).get_fdata()
            proton_data = resize(
                proton_data,
                (Im_size, Im_size, proton_data.shape[2]),
                mode='constant',
                preserve_range=True,
                order=1  # bilinear interpolation for images
            )

            mask_data = nib.load(selected_mask).get_fdata()
            mask_data = resize(
                mask_data,
                (Im_size, Im_size, mask_data.shape[2]),
                mode='constant',
                preserve_range=True,
                order=0  # nearest-neighbor for masks
            )

            # Ensure the number of slices match; trim if necessary.
            if mask_data.shape[2] != proton_data.shape[2]:
                min_slices = min(proton_data.shape[2], mask_data.shape[2])
                print(f"Warning: slice mismatch in {subject_dir}, trimming to {min_slices} slices.")
                proton_data = proton_data[..., :min_slices]
                mask_data = mask_data[..., :min_slices]

            mask_data = (mask_data > 0).astype(np.uint8)

            proton_images_list.append(proton_data)
            masks_list.append(mask_data)
            subject_paths.append(subject_dir)

        except Exception as e:
            print(f"Error loading {proton_file} or {selected_mask} in {subject_dir}: {e}")
            print("Skipping this subject.")
            continue
    
    # After processing all subjects, ensure all volumes have the same number of slices
    if proton_images_list:
        min_slices = min(vol.shape[2] for vol in proton_images_list)
        proton_images_list = [vol[..., :min_slices] for vol in proton_images_list]
        masks_list = [mask[..., :min_slices] for mask in masks_list]

    proton_images = np.array(proton_images_list, dtype=np.float32)
    masks = np.array(masks_list, dtype=np.uint8)
    return proton_images, masks, subject_paths

# ---------------------------------------------------------------------
# Other Utility Functions
# ---------------------------------------------------------------------
def save_masks(final_mask, mat_path, nifti_path, png_dir):
    savemat(mat_path, {"final_gen_mask": final_mask})
    print(f"Generated mask saved as {mat_path}.")

    nifti_img = nib.Nifti1Image(final_mask, affine=np.eye(4))
    nib.save(nifti_img, nifti_path)
    print(f"Generated mask saved as {nifti_path}.")

    os.makedirs(png_dir, exist_ok=True)
    # final_mask is shape (H, W, num_slices)
    for i in range(final_mask.shape[2]):
        plt.imsave(
            os.path.join(png_dir, f"slice_{i + 1:03}.png"),
            final_mask[:, :, i],
            cmap='gray'
        )
    print(f"Predicted slices saved as PNGs in {png_dir}.")
    
def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history.history['loss']) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, history.history['loss'], 'y', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_validation_loss.png'), dpi=300)
    plt.close()

    # IoU plot
    plt.figure()
    plt.plot(epochs, history.history['iou_score'], 'y', label='Training IoU')
    plt.plot(epochs, history.history['val_iou_score'], 'r', label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_validation_iou.png'), dpi=300)
    plt.close()
    
def visualize_image_and_mask(image, mask, title="Image and Mask"):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"{title} - Image")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(f"{title} - Mask")
    plt.imshow(mask, cmap='gray')
    # Show non-blocking and auto-close after 3 seconds
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def visualize_slice(image_volume, mask_volume, slice_idx, title_prefix=""):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"{title_prefix} - Image (Slice {slice_idx})")
    plt.imshow(image_volume[:, :, slice_idx], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(f"{title_prefix} - Mask (Slice {slice_idx})")
    plt.imshow(mask_volume[:, :, slice_idx], cmap='gray')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def visualize_augmented_samples(generator, num_samples=3):
    for _ in range(num_samples):
        img_batch, mask_batch = next(generator)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Augmented Image")
        plt.imshow(img_batch[0, :, :, 0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Augmented Mask")
        plt.imshow(mask_batch[0, :, :, 0], cmap='gray')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

def visualize_augmented_samples_overlay(generator, num_samples=3):
    for _ in range(num_samples):
        img_batch, mask_batch = next(generator)
        image = img_batch[0]           # (H, W, 3)
        mask = mask_batch[0, :, :, 0]    # (H, W)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Augmented Image")
        plt.imshow(image[..., 0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Mask Overlay")
        plt.imshow(image[..., 0], cmap='gray')
        plt.imshow(mask, alpha=0.3, cmap='Reds')  # red overlay
        plt.show(block=False)
        plt.pause(3)
        plt.close()

def plot_validation_dice(y_true, y_pred, output_dir):
    """ Quick bar plot of dice on entire validation set. """
    def dice_coef_np(y_true, y_pred, smooth=1e-6):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    dice_value = dice_coef_np(y_true, (y_pred > 0.5).astype(np.float32))
    plt.figure()
    plt.bar(['Dice Coefficient'], [dice_value], color='skyblue')
    plt.title('Validation Dice Coefficient')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'validation_dice.png'), dpi=300)
    plt.close()
    print("Validation Dice coefficient:", dice_value)

def evaluate_and_save_segmentation_plots(
    Y_true,         # shape: (N, H, W, 1)   or (N, H, W)
    Y_pred_probs,   # shape: (N, H, W, 1)   -- probabilities [0..1]
    Y_pred_bin,     # shape: (N, H, W, 1)   -- thresholded (0 or 1)
    output_dir,
    prefix="val"
):
    """
    Computes and saves:
      1) Dice distribution (boxplot)
      2) ROC curve (averaged across all slices/pixels)
      3) Confusion matrix
      4) Hausdorff distance distribution (optional, using medpy)
      5) Overlay examples of ground truth vs. predicted masks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure shapes are (N, H, W)
    if Y_true.ndim == 4:
        Y_true = Y_true[..., 0]
    if Y_pred_probs.ndim == 4:
        Y_pred_probs = Y_pred_probs[..., 0]
    if Y_pred_bin.ndim == 4:
        Y_pred_bin = Y_pred_bin[..., 0]
    
    # Dice distribution
    def dice_coef_np(y_true, y_pred, smooth=1e-6):
        intersection = np.sum(y_true * y_pred)
        return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    
    dice_per_slice = []
    for i in range(Y_true.shape[0]):
        dsc = dice_coef_np(Y_true[i], Y_pred_bin[i])
        dice_per_slice.append(dsc)
    
    plt.figure(figsize=(6, 5))
    plt.boxplot(dice_per_slice)
    plt.title("Dice Coefficient Distribution")
    plt.ylabel("DSC")
    plt.ylim([0,1])
    plt.savefig(os.path.join(output_dir, f"{prefix}_dice_boxplot.png"), dpi=300)
    plt.close()
    
    # ROC curve
    y_true_flat = Y_true.flatten()
    y_prob_flat = Y_pred_probs.flatten()

    fpr, tpr, _ = roc_curve(y_true_flat, y_prob_flat)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"{prefix}_ROC_curve.png"), dpi=300)
    plt.close()
    
    # Confusion matrix
    y_pred_flat = Y_pred_bin.flatten().astype(int)
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # Hausdorff distance (if MedPy installed)
    if USE_MEDPY:
        from medpy.metric.binary import hd
        hausdorff_list = []
        for i in range(Y_true.shape[0]):
            gt = (Y_true[i] > 0.5).astype(np.uint8)
            pr = (Y_pred_bin[i] > 0.5).astype(np.uint8)
            # If both empty, HD=0
            if gt.sum() == 0 and pr.sum() == 0:
                hausdorff_list.append(0.0)
                continue
            try:
                hdist = hd(pr, gt)
                hausdorff_list.append(hdist)
            except:
                hausdorff_list.append(np.nan)
        
        valid_hds = [h for h in hausdorff_list if not np.isnan(h)]
        if len(valid_hds) > 0:
            plt.figure(figsize=(6,5))
            plt.boxplot(valid_hds)
            plt.title("Hausdorff Distance Distribution")
            plt.ylabel("Hausdorff distance (pixels)")
            plt.savefig(os.path.join(output_dir, f"{prefix}_hausdorff_boxplot.png"), dpi=300)
            plt.close()
    else:
        print("Hausdorff distance not computed because medpy is not installed.")
    
    # Overlays
    n_examples = 5
    random_indices = np.random.choice(range(Y_true.shape[0]), size=n_examples, replace=False)
    
    overlay_dir = os.path.join(output_dir, f"{prefix}_overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    
    for i in random_indices:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        
        ax[0].set_title("Ground Truth Overlay")
        ax[0].imshow(Y_pred_probs[i], cmap='gray')
        ax[0].imshow(Y_true[i], alpha=0.3, cmap='Reds')
        ax[0].axis('off')
        
        ax[1].set_title("Prediction Overlay")
        ax[1].imshow(Y_pred_probs[i], cmap='gray')
        ax[1].imshow(Y_pred_bin[i], alpha=0.3, cmap='Reds')
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, f"overlay_{i:03d}.png"), dpi=200)
        plt.close()

