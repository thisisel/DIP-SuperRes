import argparse
import os
import torch
import requests
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from pathlib import Path
from collections import OrderedDict

from config import create_config, Config
from super_image import EdsrModel, EdsrConfig
import warnings 

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
    module="super_image.modeling_utils" 
)

def load_model(config: Config, model_arch: str) -> torch.nn.Module:
    """
    Loads the specified fine-tuned model with a definitive, robust method
    for handling all DataParallel wrapper scenarios.
    """
    print(f"Loading model architecture: {model_arch}...")
    
    # --- 1. Instantiate the base model architecture ---
    if model_arch == 'EDSR_16':
        checkpoint_path = config.model_16_block_ckpt
        model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)
    elif model_arch == 'EDSR_8':
        checkpoint_path = config.model_8_block_ckpt
        config_edsr_8block = EdsrConfig(
            scale=2,
            n_resblocks=8,
        )
        model = EdsrModel(config_edsr_8block)
    else:
        raise ValueError("Invalid model architecture specified.")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    # --- 2. Set up device and explicitly handle DataParallel for the new model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If multiple GPUs are available, wrap the model to match the likely training environment
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs. Wrapping new model in DataParallel.")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # --- 3. Load checkpoint and state dictionary ---
    print(f"Loading best model checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_state_dict = checkpoint['model_state_dict']

    # --- 4. The Definitive Reconciliation Logic ---
    is_model_parallel = isinstance(model, torch.nn.DataParallel)
    is_checkpoint_parallel = list(model_state_dict.keys())[0].startswith('module.')
    
    final_state_dict = OrderedDict()

    if is_model_parallel and not is_checkpoint_parallel:
        # SCENARIO: Current model is parallel, but checkpoint isn't. ADD "module." prefix.
        print("Model is parallel, checkpoint is not. Adding 'module.' prefix to keys...")
        for k, v in model_state_dict.items():
            final_state_dict['module.' + k] = v
    elif not is_model_parallel and is_checkpoint_parallel:
        # SCENARIO: Checkpoint is parallel, but current model isn't. STRIP "module." prefix.
        print("Checkpoint is parallel, model is not. Stripping 'module.' prefix from keys...")
        for k, v in model_state_dict.items():
            final_state_dict[k[7:]] = v
    else:
        # SCENARIO: Both are parallel or both are not. Keys match. Load directly.
        print("Model and checkpoint parallel states match. Loading directly.")
        final_state_dict = model_state_dict
        
    # --- 5. Load the correctly formatted state dictionary ---
    model.load_state_dict(final_state_dict)

    print(f"Successfully loaded model from epoch {checkpoint['epoch']}.")
    
    model.eval()
    return model, device


def load_image_from_path(image_path: str) -> Image.Image:
    """Loads an image from a local file path or a web URL."""
    if image_path.startswith(('http://', 'https://')):
        try:
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
            print(f"Successfully loaded image from URL: {image_path}")
            return image
        except requests.exceptions.RequestException as e:
            raise IOError(f"Failed to download image from URL: {e}")
    else:
        image_p = Path(image_path)
        if not image_p.exists():
            raise FileNotFoundError(f"Image file not found at: {image_path}")
        image = Image.open(image_p).convert("RGB")
        print(f"Successfully loaded image from local path: {image_path}")
        return image


def save_image(image: Image.Image, input_path: Path, output_dir: Path = None):
    """Saves the super-resolved image to the specified directory."""
    # Create a descriptive filename
    output_filename = f"{input_path.stem}_SR_{input_path.suffix}"
    
    # Determine the final output path
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / output_filename
    else:
        final_path = Path.cwd() / output_filename
        
    image.save(final_path)
    print(f"Super-resolved image saved to: {final_path}")


def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(description="Super-resolution inference script.")
    parser.add_argument(
        '--model-arch',
        required=True,
        choices=['EDSR_16', 'EDSR_8'],
        help="The model architecture to use for inference (EDSR_16 or EDSR_8)."
    )
    parser.add_argument(
        '--input-path',
        required=True,
        type=str,
        help="Path or URL to the low-resolution input image."
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help="Optional. Directory to save the output image. Defaults to the current directory."
    )
    parser.add_argument(
        '--env-mode',
        choices=['colab', 'colab-vm', 'remote', 'local'],
        default=None,
        help="Optional. Override automatic environment detection."
    )
    args = parser.parse_args()

    # 1. Setup environment and configuration
    print("--- Setting up environment ---")
    try:
        config = create_config(env_mode=args.env_mode)
        print(f"Environment configured for mode: '{config.env_mode}'")
    except Exception as e:
        print(f"Error during configuration setup: {e}")
        return

    # 2. Load the specified model
    try:
        model, device = load_model(config, args.model_arch)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading model: {e}")
        return

    # 3. Load and preprocess the input image
    try:
        lr_image = load_image_from_path(args.input_path)
        lr_tensor = ToTensor()(lr_image).unsqueeze(0).to(device)
    except (IOError, FileNotFoundError) as e:
        print(f"Error loading input image: {e}")
        return

    # 4. Run inference
    print("--- Running inference ---")
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    print("Inference complete.")

    # 5. Post-process and save the output image
    sr_image = ToPILImage()(sr_tensor.squeeze(0).cpu())
    save_image(sr_image, Path(args.input_path), args.output_dir)


if __name__ == "__main__":
    main()