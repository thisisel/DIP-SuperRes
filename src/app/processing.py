from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

from config import Config
import data
import models

try:
    from super_image.utils.metrics import compute_metrics
    from super_image.trainer_utils import EvalPrediction

except ImportError:
    print(
        "Warning: 'super_image' not found. PSNR/SSIM calculation will be unavailable."
    )

    def compute_metrics(*args, **kwargs):
        return {"psnr": 0.0, "ssim": 0.0}


@dataclass
class ProcessResult:
    """A dataclass to hold the results of the processing pipeline."""

    images: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    input_source_name: str = "Image"


def process_image_for_app(
    config: Config, model_arch: str, lr_path: str | Any, hr_path: Optional[str] = None
) -> ProcessResult:
    """
    The main orchestration function. Loads data, runs models, and computes metrics.

    Args:
        config: The application configuration object.
        model_arch: The selected model architecture ('EDSR_16' or 'EDSR_8').
        lr_path: The path or URL to the low-resolution input image.
        hr_path: Optional path to the high-resolution ground truth image.

    Returns:
        A ProcessResult object containing the final images and metrics.
    """

    if hasattr(lr_path, "name"):  # For UploadedFile objects
        input_name = lr_path.name  # type: ignore
    else:  # For Path objects
        input_name = Path(lr_path).name
    print(
        f"Processing image '{input_name}' with model '{model_arch}' in mode '{config.env_mode}'"
    )

    # 1. Load Data
    lr_image_np = data.load_img_as_np(img_src=lr_path)
    lr_tensor = data.adapt_np_as_tensor(img_raw_np=lr_image_np, is_lr=True)
    # lr_image_np = data.load_image(lr_path)
    # lr_tensor = data.normalize_for_inference(lr_image_np, is_lr=True)

    hr_tensor = None
    if hr_path:
        hr_image_np = data.load_img_as_np(img_src=hr_path)
        hr_tensor = data.adapt_np_as_tensor(img_raw_np=hr_image_np, is_lr=False)
        # hr_image_np = data.load_image(hr_path)
        # hr_tensor = data.normalize_for_inference(hr_image_np, is_lr=False)

    # 2. Load Model
    model, device = models.load_super_resolution_model(
        config, model_arch, config.env_mode
    )

    # 3. Run Inference for all models
    sr_tensor = models.run_inference(model, lr_tensor, device)
    bicubic_tensor = models.run_bicubic_interpolation(lr_tensor)

    # 4. Calculate Metrics (if in "Evaluation Mode")
    result_metrics = None
    if hr_tensor is not None:
        hr_batch = hr_tensor.unsqueeze(0).to(device)
        scale = hr_tensor.shape[1] // lr_tensor.shape[1]

        sr_metrics = compute_metrics(
            EvalPrediction(
                predictions=sr_tensor.unsqueeze(0).to(device),
                labels=hr_batch,  # type: ignore
            ),
            scale=scale,
        )
        bicubic_metrics = compute_metrics(
            EvalPrediction(
                predictions=bicubic_tensor.unsqueeze(0).to(device),
                labels=hr_batch,  # type: ignore
            ),
            scale=scale,
        )
        result_metrics = {"sr": sr_metrics, "bicubic": bicubic_metrics}

    # 5. Convert all tensors to displayable NumPy images
    result_images = {
        "lr": data.visualize_tensor(lr_tensor),
        "sr": data.visualize_tensor(sr_tensor),
        "bicubic": data.visualize_tensor(bicubic_tensor),
    }
    if hr_tensor is not None:
        result_images["hr"] = data.visualize_tensor(hr_tensor)

    return ProcessResult(
        images=result_images,
        metrics=result_metrics,
        input_source_name=input_name,
    )
