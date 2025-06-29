# MedSAM Lumbar Spine MRI Segmentation Reproduction Code (Chang et al., 2025)
# NOTE: This notebook uses the VerSe20 dataset as a public substitute to the private dataset described in the paper.
# The original dataset used in the paper is internal (150 T2-weighted axial lumbar spine MRI slices labeled by radiologists).
# This notebook compares three models: MedSAM, Original SAM, and pretrained nnUNet.
# For better alignment with the original paper, we focus only on axial slices containing visible vertebrae.
# Additionally, we filter to only include vertebrae from the lumbar region (L1-L5), using label values 51 to 55 in VerSe20.

import os
import zipfile
import requests
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# -------------------------------
# 0. Load SAM / MedSAM / nnUNet models
# -------------------------------
from segment_anything import SamPredictor, sam_model_registry
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MedSAM
CHECKPOINT_PATH_MEDSAM = "sam_vit_b_medsam.pth"  # Please place MedSAM checkpoint here
if not os.path.exists(CHECKPOINT_PATH_MEDSAM):
    raise FileNotFoundError("Please download MedSAM checkpoint and place it at 'sam_vit_b_medsam.pth'")
medsam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH_MEDSAM)
medsam.to(device=DEVICE)
predictor_medsam = SamPredictor(medsam)

# Original SAM
CHECKPOINT_PATH_SAM = "sam_vit_b.pth"  # Please place original SAM checkpoint here
if not os.path.exists(CHECKPOINT_PATH_SAM):
    raise FileNotFoundError("Please download original SAM checkpoint and place it at 'sam_vit_b.pth'")
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH_SAM)
sam.to(device=DEVICE)
predictor_sam = SamPredictor(sam)

# nnUNet pretrained inference setup
try:
    from nnunet.inference.predict import predict_from_folder
    from nnunet.paths import nnUNet_preprocessed, nnUNet_results
except ImportError:
    print("nnUNet not installed. Install with: pip install nnunet")

# Auto-run nnUNet on VerSe20 dataset if not already done
NNUNET_INPUT_DIR = "nnunet_input"
NNUNET_OUTPUT_DIR = "nnunet_preds"
TASK_NAME = "Task999_VerSe20"

# Prepare input directory for nnUNet if not exists
if not os.path.exists(NNUNET_INPUT_DIR):
    os.makedirs(NNUNET_INPUT_DIR)
    from shutil import copyfile
    raw_dir = os.path.join(DATA_DIR, "RawData")
    for fname in os.listdir(raw_dir):
        if fname.endswith(".nii.gz"):
            src = os.path.join(raw_dir, fname)
            dst = os.path.join(NNUNET_INPUT_DIR, fname)
            copyfile(src, dst)

# Run nnUNet inference (only if results not already present)
if not os.path.exists(NNUNET_OUTPUT_DIR) or len(os.listdir(NNUNET_OUTPUT_DIR)) == 0:
    print("Running nnUNet inference on VerSe20...")
    predict_from_folder(
        model_training_output_dir=f"results/{TASK_NAME}/nnUNetTrainer__nnUNetPlans__3d_fullres",
        input_folder=NNUNET_INPUT_DIR,
        output_folder=NNUNET_OUTPUT_DIR,
        folds=[0],
        save_npz=False,
        num_threads_preprocessing=2,
        num_threads_nifti_save=2
    )

# -------------------------------
# Updated working public dataset: https://zenodo.org/record/5213867 (VerSe 20 training set)
DATASET_URL = "https://zenodo.org/record/5213867/files/VerSe_20_training.zip?download=1"
DATASET_ZIP = "VerSe_20_training.zip"
DATA_DIR = "verse2020"

if not os.path.exists(DATASET_ZIP):
    print("Downloading dataset...")
    with requests.get(DATASET_URL, stream=True) as r:
        with open(DATASET_ZIP, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

if not os.path.exists(DATA_DIR):
    print("Extracting dataset...")
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

# -------------------------------
# 2. Helper Functions
# -------------------------------
def is_lumbar_slice(mask_slice):
    """
    Check if a given 2D mask slice contains any lumbar vertebrae (labels 51 to 55).
    """
    lumbar_labels = [51, 52, 53, 54, 55]
    return np.isin(mask_slice, lumbar_labels).any()

# This function should be called in the evaluation loop to skip slices that do not contain L1–L5 vertebrae.

# Inserted directly into main evaluation loop
results = {"MedSAM": [], "SAM": [], "nnUNet": []}

for patient_file in os.listdir(os.path.join(DATA_DIR, "VertebraeSegmentation")):
    if not patient_file.endswith("_seg.nii.gz"):
        continue
    gt_path = os.path.join(DATA_DIR, "VertebraeSegmentation", patient_file)
    img_path = gt_path.replace("_seg.nii.gz", ".nii.gz").replace("VertebraeSegmentation", "RawData")

    gt_mask = nib.load(gt_path).get_fdata()
    image = nib.load(img_path).get_fdata()

    for slice_idx in range(image.shape[2]):
        gt_mask_slice = gt_mask[:, :, slice_idx]
        if not is_lumbar_slice(gt_mask_slice):
            continue  # Skip slices without lumbar vertebrae

        img_slice = image[:, :, slice_idx].astype(np.float32)
        img_norm = (img_slice - img_slice.min()) / (img_slice.ptp() + 1e-6)
        img_rgb = np.stack([img_norm]*3, axis=-1) * 255
        img_rgb = img_rgb.astype(np.uint8)

        # Bounding box prompt from GT mask
        ys, xs = np.where(gt_mask_slice > 0)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        input_box = np.array([x0, y0, x1, y1])

        # Predict with MedSAM
        predictor_medsam.set_image(img_rgb)
        medsam_masks, _, _ = predictor_medsam.predict(box=input_box[None, :], return_logits=False)
        pred_mask_medsam = medsam_masks[0].astype(np.uint8)

        # Predict with Original SAM
        predictor_sam.set_image(img_rgb)
        sam_masks, _, _ = predictor_sam.predict(box=input_box[None, :], return_logits=False)
        pred_mask_sam = sam_masks[0].astype(np.uint8)

        # Ground truth binarized
        gt_bin = (gt_mask_slice > 0).astype(np.uint8)

        def dice(pred, gt):
            return 2 * np.sum(pred * gt) / (np.sum(pred) + np.sum(gt) + 1e-6)

        def iou(pred, gt):
            return np.sum(pred * gt) / (np.sum((pred + gt) > 0) + 1e-6)

        results["MedSAM"].append((dice(pred_mask_medsam, gt_bin), iou(pred_mask_medsam, gt_bin)))
        results["SAM"].append((dice(pred_mask_sam, gt_bin), iou(pred_mask_sam, gt_bin)))

        # nnUNet prediction
        patient_id = patient_file.replace("_seg.nii.gz", "")
        nnunet_mask_path = os.path.join("nnunet_preds", f"{patient_id}_seg.nii.gz")
        if os.path.exists(nnunet_mask_path):
            nnunet_mask = nib.load(nnunet_mask_path).get_fdata()
            pred_mask_nnunet = (nnunet_mask[:, :, slice_idx] > 0).astype(np.uint8)
            results["nnUNet"].append((dice(pred_mask_nnunet, gt_bin), iou(pred_mask_nnunet, gt_bin)))

# Compute averages
for model in results:
    if len(results[model]) > 0:
        dices, ious = zip(*results[model])
        print(f"{model}: Avg Dice = {np.mean(dices):.4f}, Avg IoU = {np.mean(ious):.4f}")
[... unchanged remainder of code ...]
