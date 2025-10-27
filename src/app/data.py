import numpy as np
from numpy.typing import NDArray
import torch
import requests
from PIL import Image
from pathlib import Path
from torchvision.transforms import ToTensor, ToPILImage
from typing import List, Any, Union
import io

from config import Config, LR_MAX_PER_CHANNEL, HR_MAX_PER_CHANNEL


def load_adapt_image(image_path: str | Path) -> np.ndarray:
    """
    Loads a standard image, enforces the data contract regarding
    channel order (BGR) and data type/range (uint16).
    """
    img_pil = Image.open(image_path).convert("RGB")
    img_np_rgb = np.array(img_pil)

    # Rescale uint8 (0-255) to uint16 (0-65535) dynamic range
    if img_np_rgb.dtype == np.uint8:
        img_np_rgb = (img_np_rgb.astype(np.float32) / 255.0) * 65535.0

    img_np_uint16_rgb = img_np_rgb.astype(np.uint16)

    # Convert from RGB to BGR to match the training pipeline
    img_np_uint16_bgr = img_np_uint16_rgb[..., ::-1].copy()

    return img_np_uint16_bgr


def load_img_as_np(img_src: Union[str, Path, io.BytesIO]) -> np.ndarray:
    """
    Loads an image from a URL, local path, or file-like object and converts it to a numpy array.

    Args:
        img_src: The source of the image, which can be a URL string, a Path object,
                 or a file-like object (like those from st.file_uploader).

    Returns:
        A numpy array representing the image.

    Raises:
        IOError: If the image fails to load.
        FileNotFoundError: If the file is not found at the specified path.
    """
    try:
        # Case 1: Handle URL string
        if isinstance(img_src, str) and img_src.startswith(("http://", "https://")):
            response = requests.get(img_src, stream=True)
            response.raise_for_status()
            img = Image.open(response.raw)

        # Case 2: Handle Path object or local string path
        elif isinstance(img_src, (str, Path)):
            img_path = Path(img_src)
            if not img_path.exists() or not img_path.is_file():
                raise FileNotFoundError(f"Image file not found at: {img_path}")

            if img_path.suffix.lower() == ".npy":
                return np.load(img_path)

            img = Image.open(img_path)

        # Case 3: Handle file-like objects 
        elif hasattr(img_src, "read"):
            img = Image.open(img_src)

        else:
            raise TypeError("Unsupported image source type.")

        return np.array(img.convert("RGB"))

    except Exception as e:
        raise IOError(f"Failed to load image from source: {e}")


def adapt_np_as_tensor(img_raw_np: np.ndarray, is_lr: bool):
    orig_dtype = img_raw_np.dtype
    if orig_dtype == np.uint16:
        LR_MAX_TENSOR = torch.tensor(LR_MAX_PER_CHANNEL, dtype=torch.float32).view(
            3, 1, 1
        )
        HR_MAX_TENSOR = torch.tensor(HR_MAX_PER_CHANNEL, dtype=torch.float32).view(
            3, 1, 1
        )
        max_vals = LR_MAX_TENSOR if is_lr else HR_MAX_TENSOR
        tensor_bgr = torch.from_numpy(img_raw_np.astype(np.float32)).permute(2, 0, 1)
        normalized_tensor = tensor_bgr / max_vals
    elif orig_dtype == np.uint8:
        if img_raw_np.shape[2] == 3:  # Check if it's a color image
            raw_np_bgr = img_raw_np[..., ::-1].copy()
        else:
            raw_np_bgr = img_raw_np

        tensor_bgr = torch.from_numpy(raw_np_bgr.astype(np.float32)).permute(2, 0, 1)
        normalized_tensor = tensor_bgr / 255.0
    else:
        raise TypeError(f"Unsupported input image dtype: {orig_dtype}")

    return torch.clamp(normalized_tensor, 0.0, 1.0)


def load_image(image_source: str | Any) -> NDArray:
    """Loads either a PIL image from a local file path / a web URL
    or a prepared sample from npy stored array.
    """
    if hasattr(image_source, "read"):
        try:
            image_np_adapted = load_adapt_image(image_source)
        except Exception as e:
            raise IOError(f"Failed to read image from file-like object: {e}")
    elif str(image_source).startswith(("http://", "https://")):
        try:
            response = requests.get(str(image_source), stream=True)
            response.raise_for_status()
            image_np_adapted = load_adapt_image(response.raw)  # type: ignore

        except requests.exceptions.RequestException as e:
            raise IOError(f"Failed to download image from URL: {e}")
    else:
        p = Path(image_source)
        if not (p.exists() and p.is_file()):
            raise FileNotFoundError(f"Image file not found at: {p}")
        image_np_adapted = (
            np.load(p) if p.suffix.lower() == ".npy" else load_adapt_image(p)
        )

    return image_np_adapted


def get_example_image_paths(config: Config) -> List[Path]:
    if not config.example_lr_dir.exists():
        return []
    return sorted(list(config.example_lr_dir.glob("*.npy")))


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return ToTensor()(image)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    return ToPILImage()(tensor.squeeze(0).cpu())


def normalize_for_inference(raw_bgr_np: np.ndarray, is_lr: bool) -> torch.Tensor:
    """
    Takes a raw uint16 BGR NumPy array (H, W, C) and performs the final
    normalization to create a model-ready tensor.
    """
    # Convert to float32 tensor and permute to (C, H, W)
    tensor_bgr = torch.from_numpy(raw_bgr_np.astype(np.float32)).permute(2, 0, 1)

    # Apply GLOBAL Min-Max normalization
    LR_MAX_TENSOR = torch.tensor(LR_MAX_PER_CHANNEL, dtype=torch.float32).view(3, 1, 1)
    HR_MAX_TENSOR = torch.tensor(HR_MAX_PER_CHANNEL, dtype=torch.float32).view(3, 1, 1)
    max_vals = LR_MAX_TENSOR if is_lr else HR_MAX_TENSOR
    normalized_tensor = tensor_bgr / max_vals

    return torch.clamp(normalized_tensor, 0.0, 1.0)


def visualize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Converts a tensor to a displayable NumPy array with contrast stretching."""
    image = tensor.cpu().detach().numpy()
    vmin = np.percentile(image, 2, axis=(1, 2), keepdims=True)
    vmax = np.percentile(image, 98, axis=(1, 2), keepdims=True)
    image = np.clip(image, vmin, vmax)
    image = (image - vmin) / (
        vmax - vmin + 1e-6
    )  # Add epsilon to avoid division by zero
    image = np.transpose(image, (1, 2, 0))
    return np.clip(image, 0, 1)
