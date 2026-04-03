from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None


def otsu_brain_mask(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image_u8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        image_u8 = image.copy()

    _, thresh = cv2.threshold(image_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    labeled, num = ndi.label(cleaned > 0)
    if num == 0:
        return (image_u8 > 0).astype(np.uint8)

    sizes = ndi.sum(np.ones_like(labeled), labeled, index=np.arange(1, num + 1))
    largest_component = np.argmax(sizes) + 1
    mask = (labeled == largest_component).astype(np.uint8)
    return mask


def n4_bias_correct(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if sitk is None:
        return image

    image_float = image.astype(np.float32)
    itk_image = sitk.GetImageFromArray(image_float)

    if mask is None:
        mask = (image_float > 0).astype(np.uint8)
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(itk_image, itk_mask)
    corrected_arr = sitk.GetArrayFromImage(corrected)
    return corrected_arr.astype(np.float32)


def zscore_normalize(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mask = mask.astype(bool)
    if mask.sum() == 0:
        mean = image.mean()
        std = image.std() + 1e-8
        return (image - mean) / std

    region = image[mask]
    mean = region.mean()
    std = region.std() + 1e-8
    normalized = (image - mean) / std
    return normalized.astype(np.float32)


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    brain_mask: np.ndarray,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    brain_resized = cv2.resize(brain_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return image_resized, mask_resized, brain_resized


def preprocess_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    image_size: int = 256,
    skull_strip: bool = True,
    n4_bias_correction: bool = True,
    zscore: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = image.astype(np.float32)
    mask = (mask > 0).astype(np.uint8)

    if skull_strip:
        brain_mask = otsu_brain_mask(image)
    else:
        brain_mask = (image > 0).astype(np.uint8)

    if n4_bias_correction:
        image = n4_bias_correct(image, brain_mask)

    image = image * brain_mask.astype(np.float32)

    if zscore:
        image = zscore_normalize(image, brain_mask)

    image, mask, brain_mask = resize_image_and_mask(image, mask, brain_mask, image_size)
    return image.astype(np.float32), mask.astype(np.uint8), brain_mask.astype(np.uint8)
