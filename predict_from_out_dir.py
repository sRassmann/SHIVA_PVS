# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx

import gc
import os
import time
import numpy as np
from pathlib import Path
import argparse
import nibabel
import tensorflow as tf
import torch
from monai import transforms
from glob import glob
from tqdm import tqdm

size = (160, 214, 176)


@torch.no_grad()
def close_mask(mask, dilate_size=7, erode_size=5):
    org_shape, org_dtype = mask.shape, mask.dtype

    # dilate
    filter = (
        torch.ones((dilate_size, dilate_size, dilate_size)).unsqueeze(0).unsqueeze(0)
    )
    conv_res = torch.nn.functional.conv3d(
        mask.unsqueeze(0), filter, padding=dilate_size // 2
    )
    dil = conv_res > 0

    # erode
    filter = torch.ones((erode_size, erode_size, erode_size)).unsqueeze(0).unsqueeze(0)
    conv_res = torch.nn.functional.conv3d(
        dil.float(), filter, padding=erode_size // 2
    ).squeeze(dim=0)
    erode = conv_res == filter.sum()

    assert erode.shape == org_shape

    return erode.to(org_dtype)


def is_greater_0(x):
    return x > 0


def get_transforms(args):
    load_transforms = transforms.Compose(
        [
            transforms.LoadImaged(
                image_only=True, ensure_channel_first=True, keys=["t1", "flair", "mask"]
            ),
            transforms.Lambdad(keys="mask", func=close_mask),
            transforms.MaskIntensityd(
                keys=["t1", "flair"],
                mask_key="mask",
                allow_missing_keys=False,
                select_fn=is_greater_0,
            )
            if args.skull_strip
            else transforms.Identityd(keys="mask"),
            transforms.CropForegroundd(
                keys=("mask", "t1", "flair"),
                source_key="mask",
                allow_smaller=True,
                margin=1,
            ),
            # based on this comment: https://github.com/pboutinaud/SHIVA_PVS/issues/2#issuecomment-1499029783 and what they mention in the paper
            transforms.Orientationd(keys=["mask", "flair", "t1"], axcodes="RAS"),
            transforms.ResizeWithPadOrCropD(
                spatial_size=size, keys=["mask", "flair", "t1"]
            ),
        ]
    )
    return load_transforms


def predict_image(
    flair,
    t1,
    output_path,
    mask_path,
    predictor_files,
    t,
    save_original=False,
    verbose=True,
    threshold=0.5,
):
    img = {"flair": flair, "t1": t1, "mask": mask_path}
    img = t(img)

    images = []
    for mod in ["t1", "flair"]:
        image = img[mod].numpy()
        image /= np.percentile(image, 99)
        image = np.reshape(image, (1, *size, 1))
        images.append(image)
    input_image = np.concatenate(images, axis=-1)
    mask = img["mask"].numpy()
    predictions = []
    for predictor_file in predictor_files:
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            model = tf.keras.models.load_model(
                predictor_file, compile=False, custom_objects={"tf": tf}
            )
        except Exception as err:
            print(f"\n\tWARNING : Exception loading model : {predictor_file}\n{err}")
            continue
        prediction = model.predict(input_image, batch_size=1)
        if verbose:
            print(prediction.sum())
        predictions.append(prediction)
    # Average all predictions
    predictions = np.stack(predictions, axis=0)[..., 0]

    predictions = np.mean(predictions, axis=0) * (mask > 0)
    if threshold > 0:
        predictions = (predictions > threshold).astype(np.uint8)
    img["mask"] *= 0
    img["mask"] += torch.Tensor(predictions)
    inv = t.inverse(img)
    affine = inv["mask"].meta["affine"].numpy().squeeze().astype(np.float32)
    # Save prediction
    nifti = nibabel.Nifti1Image(
        inv["mask"].squeeze(dim=0).astype(np.uint8), affine=affine
    )
    nibabel.save(nifti, output_path)
    if save_original:
        nifti = nibabel.Nifti1Image(
            (inv["image"] * 255).squeeze(dim=0).astype(np.float32), affine=affine
        )
        org_out = output_path.replace(".nii.gz", "_input.nii.gz")
        print(f"Saving original image to {org_out}")
        nibabel.save(nifti, org_out)


def main(args):
    # The tf model files for the predictors, the prediction will be averaged
    predictor_files = args.model
    if not predictor_files:
        predictor_files = [
            f"PVS/v1-T1-FLAIR.PVS/20220223-182702_Unet3Dv2-10.7.2-1.8-T1_FLAIR.VRS_fold_VRS_1x5_fromMH_fold_{i}_model.h5"
            for i in range(5)
        ]
        print(
            f"Using default model ensemble from {os.path.dirname(predictor_files[0])}"
        )

    print(os.path.join(args.input, "*", args.file))
    files = glob(os.path.join(args.input, "*", args.file))
    print(f"Found {len(files)} files in {args.input}.")
    for input_image in tqdm(files):
        predict_image(
            input_image,
            os.path.join(os.path.dirname(input_image), "t1.nii.gz"),
            input_image.replace(".nii.gz", "_pvs_seg.nii.gz"),
            os.path.join(os.path.dirname(input_image), "mask.nii.gz"),
            predictor_files,
            get_transforms(args),
            save_original=args.save_original,
            verbose=False,
        )


if __name__ == "__main__":
    # Script parameters
    parser = argparse.ArgumentParser(
        description="Run inference with tensorflow models(s) on an image that may be built from several modalities"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="input directory containing the subjects",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="name of input file within subject directory",
        default="pred_flair.nii.gz",
    )
    parser.add_argument(
        "-m", "--model", type=Path, action="append", help="(multiple) prediction models"
    )
    parser.add_argument(
        "--save_original",
        action="store_true",
        help="Save the original image to the output directory",
    )
    parser.add_argument(
        "-s",
        "--skull_strip",
        action="store_true",
        help="Skull strip the image before inference",
    )

    args = parser.parse_args()

    main(args)
