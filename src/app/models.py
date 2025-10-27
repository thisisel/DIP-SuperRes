import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from collections import OrderedDict
from super_image import EdsrModel, EdsrConfig

from config import Config
from data import pil_to_tensor, tensor_to_pil

# A simple mock model class for local CPU-only development
class MockModel:
    def __call__(self, tensor_image: torch.Tensor) -> torch.Tensor:
        pil_image = tensor_to_pil(tensor_image)
        # Simulate an SR operation with a simple sharpening filter
        mock_sr_image = pil_image.filter(ImageFilter.SHARPEN)
        return pil_to_tensor(mock_sr_image)

def load_super_resolution_model(config: Config, model_arch: str, mode: str):
    if mode == "local-mock":
        print("Loading MOCK model for local development.")
        return MockModel(), torch.device("cpu")
    
    # --- Real Model Loading Logic ---
    print(f"Loading REAL model: {model_arch}")
    if model_arch == 'EDSR_16':
        checkpoint_path = config.model_16_block_ckpt
        model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2, n_resblocks=16)
    else: # EDSR_8
        checkpoint_path = config.model_8_block_ckpt
        config_edsr_8block = EdsrConfig(
            scale=2,
            n_resblocks=8,
        )
        model = EdsrModel(config_edsr_8block)
        
    config.validate_for_inference(model_arch)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs. Wrapping new model in DataParallel.")
        model = torch.nn.DataParallel(model)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_state_dict = checkpoint['model_state_dict']

    is_model_parallel = isinstance(model, torch.nn.DataParallel)
    is_checkpoint_parallel = list(model_state_dict.keys())[0].startswith('module.')
    
    final_state_dict = OrderedDict()

    if is_model_parallel and not is_checkpoint_parallel:
        # ADD "module." prefix.
        print("Model is parallel, checkpoint is not. Adding 'module.' prefix to keys...")
        for k, v in model_state_dict.items():
            final_state_dict['module.' + k] = v
    elif not is_model_parallel and is_checkpoint_parallel:
        # STRIP "module." prefix.
        print("Checkpoint is parallel, model is not. Stripping 'module.' prefix from keys...")
        for k, v in model_state_dict.items():
            final_state_dict[k[7:]] = v
    else:
        #  Keys match -> Load directly.
        print("Model and checkpoint parallel states match. Loading directly.")
        final_state_dict = model_state_dict
        
    # --- 5. Load the correctly formatted state dictionary ---
    model.load_state_dict(final_state_dict)
        
    print(f"Successfully loaded model from epoch {checkpoint['epoch']}.")
    # model.to(device)
    model.eval()
    return model, device

def run_inference(model, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(model, MockModel):
        return model(image_tensor)
    
    with torch.no_grad():
        input_batch = image_tensor.unsqueeze(0).to(device)
        output_tensor = model(input_batch)
        return output_tensor.squeeze(0)

def run_bicubic_interpolation(lr_tensor: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
    lr_height, lr_width = lr_tensor.shape[1], lr_tensor.shape[2]
    hr_height, hr_width = lr_height * scale_factor, lr_width * scale_factor
    
    return F.interpolate(
        lr_tensor.unsqueeze(0),
        size=(hr_height, hr_width),
        mode='bicubic',
        align_corners=False
    ).squeeze(0)