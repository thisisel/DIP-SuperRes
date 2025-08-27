import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    pass

EnvModeType = Literal["colab", "remote", "local"]


@dataclass(frozen=True)
class Config:
    env_mode: EnvModeType
    # Optional override fields for user-defined paths
    model_16_block_ckpt_override: Optional[Path] = field(default=None)
    model_8_block_ckpt_override: Optional[Path] = field(default=None)

    # Derived fields (init=False)
    data_root: Path = field(init=False)
    base_dir: Path = field(init=False)
    finetune_dir: Path = field(init=False)

    # --- Final checkpoint paths that the inference script will use ---
    model_16_block_ckpt: Path = field(init=False)
    model_8_block_ckpt: Path = field(init=False)

    def __post_init__(self) -> None:
        data_root_map = {
            "local": Path("/mnt/shared"),
            "remote": Path.home(),
            "colab": Path("/content/drive/MyDrive"),
            "colab-vm": Path("/content/MyDrive"),
        }
        object.__setattr__(
            self, "data_root", data_root_map.get(self.env_mode, Path.cwd())
        )
        object.__setattr__(self, "base_dir", self.data_root / "datasets/sen2venus")
        object.__setattr__(self, "finetune_dir", self.base_dir / "finetune")

        # Logic to prioritize .env paths over defaults
        # For 16-block model
        if self.model_16_block_ckpt_override:
            object.__setattr__(
                self, "model_16_block_ckpt", self.model_16_block_ckpt_override
            )
        else:
            default_path_16 = (
                self.finetune_dir / "edsr_base" / "best_model_checkpoint.pt"
            )
            object.__setattr__(self, "model_16_block_ckpt", default_path_16)

        # For 8-block model
        if self.model_8_block_ckpt_override:
            object.__setattr__(
                self, "model_8_block_ckpt", self.model_8_block_ckpt_override
            )
        else:
            default_path_8 = (
                self.finetune_dir / "edsr_base_8_block" / "best_model_checkpoint.pt"
            )
            object.__setattr__(self, "model_8_block_ckpt", default_path_8)

    def validate(self) -> None:
        """Validate config paths exist; raise errors otherwise."""
        paths_to_check = {
            # "data_root": self.data_root,
            # "base_dir": self.base_dir,
            "16-block checkpoint": self.model_16_block_ckpt,
            "8-block checkpoint": self.model_8_block_ckpt,
        }
        missing = [
            f"{name} ({path})"
            for name, path in paths_to_check.items()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing required paths/files: {', '.join(missing)}"
            )


def setup_environment(env_mode: EnvModeType) -> None:
    """Perform environment-specific setup."""
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


def create_config(env_mode: EnvModeType | None = None) -> Config:
    """Factory to create and setup config based on detected environment."""
    if env_mode is None:
        if "google.colab" in sys.modules:
            env_mode = "colab"
        else:
            env_mode = "local" 

    # Load environment variables from a .env file if it exists
    # This should be called after setup to ensure dotenv is installed
    setup_environment(env_mode)
    load_dotenv()

    # --- Read checkpoint paths from environment variables ---
    ckpt_16_path_str = os.getenv("CKPT_PATH_EDSR_16")
    ckpt_8_path_str = os.getenv("CKPT_PATH_EDSR_8")

    # Convert to Path objects if they exist
    ckpt_16_path = Path(ckpt_16_path_str) if ckpt_16_path_str else None
    ckpt_8_path = Path(ckpt_8_path_str) if ckpt_8_path_str else None

    config = Config(
        env_mode=env_mode,
        model_16_block_ckpt_override=ckpt_16_path,
        model_8_block_ckpt_override=ckpt_8_path,
    )
    config.validate()
    return config
