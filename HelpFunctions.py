import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import nibabel as nib
from scipy.io import savemat
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import Sequence

try:
    from medpy.metric.binary import hd
    USE_MEDPY = True
except ImportError:
    print("medpy not installed; Hausdorff distance will be skipped.")
    USE_MEDPY = False

# ---------------------------------------------------------------------
# Data Generator for loading NIfTI volumes using Keras Sequence
# ---------------------------------------------------------------------
class NiftiSequence(Sequence):
    def __init__(self, subject_dirs, batch_size, image_size, 
                 img_aug=None, mask_aug=None, augment=False, shuffle=True):
        """
        subject_dirs: list of directories, one per subject
        batch_size: number of subjects per batch (subject-level batching)
        image_size: target image size (assumed square)
        img_aug: dictionary of image augmentation parameters (from config)
        mask_aug: dictionary of mask augmentation parameters (from config)
        augment: Boolean flag to apply augmentation or not
        shuffle: whether to shuffle the order at the end of each epoch
        """
        self.subject_dirs = subject_dirs
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.subject_dirs))
        self.on_epoch_end()
        
        if self.augment:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            self.img_datagen = ImageDataGenerator(**(img_aug if img_aug else {}))
            self.mask_datagen = ImageDataGenerator(**(mask_aug if mask_aug else {}))
    
    def __len__(self):
        return int(np.ceil(len(self.subject_dirs) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_subjects = [self.subject_dirs[k] for k in batch_indexes]
        X, Y = self.__data_generation(batch_subjects)
        return X, Y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_subjects):
        X_batch = []
        Y_batch = []
        
        # Define the maximum number of slices per subject to load.
        # Adjust this number as needed.
        max_slices_per_subject = 16

        for subject_dir in batch_subjects:
            proton_file = self._find_proton_file(subject_dir)
            mask_file = self._find_mask_file(subject_dir)
            if proton_file is None or mask_file is None:
                continue
            
            # Load volumes and force them to be float32 to save memory.
            proton_data = nib.load(proton_file).get_fdata().astype(np.float32)
            mask_data = nib.load(mask_file).get_fdata().astype(np.float32)
            
            # Resize volumes to target image size (and keep all slices for now)
            proton_data = resize(proton_data, (self.image_size, self.image_size, proton_data.shape[2]),
                                mode='constant', preserve_range=True, order=1)
            mask_data = resize(mask_data, (self.image_size, self.image_size, mask_data.shape[2]),
                            mode='constant', preserve_range=True, order=0)
            
            # Ensure both volumes have the same number of slices
            num_slices = min(proton_data.shape[2], mask_data.shape[2])
            proton_data = proton_data[..., :num_slices]
            mask_data = mask_data[..., :num_slices]
            
            # Randomly sample a fixed number of slices (if there are more than max_slices_per_subject)
            if num_slices > max_slices_per_subject:
                slice_indices = np.sort(np.random.choice(num_slices, max_slices_per_subject, replace=False))
            else:
                slice_indices = np.arange(num_slices)
            
            # Expand dims for channel (if not already)
            proton_data = np.expand_dims(proton_data, axis=-1)
            mask_data = np.expand_dims(mask_data, axis=-1)
            
            # Process only the selected slices
            for i in slice_indices:
                X_slice = proton_data[:, :, i, :]
                Y_slice = (mask_data[:, :, i, :] > 0).astype(np.uint8)
                
                if self.augment:
                    transform = self.img_datagen.get_random_transform(X_slice.shape)
                    X_slice = self.img_datagen.apply_transform(X_slice, transform)
                    Y_slice = self.mask_datagen.apply_transform(Y_slice, transform)
                
                X_batch.append(X_slice)
                Y_batch.append(Y_slice)
        
        if len(X_batch) == 0:
            return (np.empty((0, self.image_size, self.image_size, 3)),
                    np.empty((0, self.image_size, self.image_size, 1)))
        
        X_batch = np.stack(X_batch, axis=0)
        Y_batch = np.stack(Y_batch, axis=0)
        # Replicate the single channel to 3 channels for images.
        X_batch = np.repeat(X_batch, 3, axis=-1)
        X_batch = X_batch.astype('float32') / 255.0
        Y_batch = Y_batch.astype('float32')
        return X_batch, Y_batch

    def _find_proton_file(self, subject_dir):
        import glob
        candidates = glob.glob(os.path.join(subject_dir, '*[Pp]roton*.*nii*'))
        return candidates[0] if candidates else None
    
    def _find_mask_file(self, subject_dir):
        import glob
        candidates = glob.glob(os.path.join(subject_dir, '*[Mm]ask*.*nii*'))
        for candidate in candidates:
            if os.path.basename(candidate).lower() == "mask.nii":
                return candidate
        return candidates[0] if candidates else None

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
    for i in range(final_mask.shape[2]):
        plt.imsave(os.path.join(png_dir, f"slice_{i+1:03}.png"), final_mask[:, :, i], cmap='gray')
    print(f"Predicted slices saved as PNGs in {png_dir}.")

def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure()
    plt.plot(epochs, history.history['loss'], 'y', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_validation_loss.png'), dpi=300)
    plt.close()
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
        # Instead of next(), use __getitem__ with a random index
        idx = np.random.randint(0, len(generator))
        img_batch, mask_batch = generator[idx]
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
        idx = np.random.randint(0, len(generator))
        img_batch, mask_batch = generator[idx]
        image = img_batch[0]  # (H, W, 3)
        mask = mask_batch[0, :, :, 0]  # (H, W)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Augmented Image")
        plt.imshow(image[..., 0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Mask Overlay")
        plt.imshow(image[..., 0], cmap='gray')
        plt.imshow(mask, alpha=0.3, cmap='Reds')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

def plot_validation_dice(y_true, y_pred, output_dir):
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

def evaluate_and_save_segmentation_plots(Y_true, Y_pred_probs, Y_pred_bin, output_dir, prefix="val"):
    os.makedirs(output_dir, exist_ok=True)
    if Y_true.ndim == 4:
        Y_true = Y_true[..., 0]
    if Y_pred_probs.ndim == 4:
        Y_pred_probs = Y_pred_probs[..., 0]
    if Y_pred_bin.ndim == 4:
        Y_pred_bin = Y_pred_bin[..., 0]
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
    plt.ylim([0, 1])
    plt.savefig(os.path.join(output_dir, f"{prefix}_dice_boxplot.png"), dpi=300)
    plt.close()
    
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
    
    y_pred_flat = Y_pred_bin.flatten().astype(int)
    cm = confusion_matrix(Y_true.flatten(), y_pred_flat, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"), dpi=300)
    plt.close()
    
    if USE_MEDPY:
        from medpy.metric.binary import hd
        hausdorff_list = []
        for i in range(Y_true.shape[0]):
            gt = (Y_true[i] > 0.5).astype(np.uint8)
            pr = (Y_pred_bin[i] > 0.5).astype(np.uint8)
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
            plt.figure(figsize=(6, 5))
            plt.boxplot(valid_hds)
            plt.title("Hausdorff Distance Distribution")
            plt.ylabel("Hausdorff distance (pixels)")
            plt.savefig(os.path.join(output_dir, f"{prefix}_hausdorff_boxplot.png"), dpi=300)
            plt.close()
    else:
        print("Hausdorff distance not computed because medpy is not installed.")
    
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
