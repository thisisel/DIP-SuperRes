import os
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv

EnvModeType = Literal["colab", "colab-vm", "remote", "local", "local-mock"]

LR_MAX_PER_CHANNEL = [8683.0, 9235.0, 11554.0]
HR_MAX_PER_CHANNEL = [11557.0, 11554.0, 12518.0]


@dataclass(frozen=True)
class Config:
    env_mode: EnvModeType
    model_16_block_ckpt_override: Optional[Path] = field(default=None)
    model_8_block_ckpt_override: Optional[Path] = field(default=None)

    # Derived fields (init=False)
    data_root: Path = field(init=False)
    base_dir: Path = field(init=False)
    assets_dir: Path = field(init=False)
    example_lr_dir: Path = field(init=False)
    example_hr_dir: Path = field(init=False)
    finetune_dir: Path = field(init=False)

    # --- Final checkpoint paths that the inference script will use ---
    model_16_block_ckpt: Path = field(init=False)
    model_8_block_ckpt: Path = field(init=False)

    def __post_init__(self) -> None:
        data_root_map = {
            "local": Path.home(),
            "local-mock": Path.home(),
            "remote": Path.home(),
            "colab": Path("/content/drive/MyDrive"),
        }
        object.__setattr__(
            self, "data_root", data_root_map.get(self.env_mode, Path.cwd())
        )
        object.__setattr__(
            self, "base_dir", self.data_root / "datasets/sen2venus"
        ) 
        object.__setattr__(self, "assets_dir", Path.cwd() / "assets")
        object.__setattr__(
            self, "example_lr_dir", self.assets_dir / "examples_npy" / "lr"
        )
        object.__setattr__(
            self, "example_hr_dir", self.assets_dir / "examples_npy" / "hr"
        )
        object.__setattr__(self, "finetune_dir", self.base_dir / "finetune")

        # Logic to prioritize .env paths over defaults
        if self.model_16_block_ckpt_override:
            object.__setattr__(
                self, "model_16_block_ckpt", self.model_16_block_ckpt_override
            )
        else:
            object.__setattr__(
                self,
                "model_16_block_ckpt",
                self.finetune_dir / "edsr_base" / "best_model_checkpoint.pt",
            )

        if self.model_8_block_ckpt_override:
            object.__setattr__(
                self, "model_8_block_ckpt", self.model_8_block_ckpt_override
            )
        else:
            object.__setattr__(
                self,
                "model_8_block_ckpt",
                self.finetune_dir / "edsr_base_8_block" / "best_model_checkpoint.pt",
            )

    def validate_for_inference(self, model_arch: str):
        paths_to_check = {"assets_dir": self.assets_dir}
        if model_arch == "EDSR_16":
            paths_to_check["16-block checkpoint"] = self.model_16_block_ckpt
        elif model_arch == "EDSR_8":
            paths_to_check["8-block checkpoint"] = self.model_8_block_ckpt

        missing = [
            f"{name} ({path})"
            for name, path in paths_to_check.items()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing required paths/files for inference: {', '.join(missing)}"
            )


def setup_environment(env_mode: EnvModeType) -> None:
    if env_mode is None:
        if "google.colab" in sys.modules:
            env_mode = "colab"
        else:
            env_mode = "local"

    if env_mode.startswith("colab"):
        packages = ["super-image", "python-dotenv"]
        try:
            for package in packages:
                __import__(package)
        except ImportError:
            print("Installing external packages...")
            subprocess.run(["pip", "install", "--quiet"] + packages, check=True)

        from google.colab import drive

        drive.mount("/content/drive", force_remount=True)


def create_config(env_mode: EnvModeType) -> Config:
    env_mode = (
        "colab" if env_mode is None and "google.colab" in sys.modules else "local"
    )
    load_dotenv()
    ckpt_16_path = Path(p) if (p := os.getenv("CKPT_PATH_EDSR_16")) else None
    ckpt_8_path = Path(p) if (p := os.getenv("CKPT_PATH_EDSR_8")) else None

    config = Config(
        env_mode=env_mode,
        model_16_block_ckpt_override=ckpt_16_path,
        model_8_block_ckpt_override=ckpt_8_path,
    )
    return config
