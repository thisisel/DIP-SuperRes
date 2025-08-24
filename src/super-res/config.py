import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

EnvModeType = Literal["colab", "colab-vm", "remote", "local"]


@dataclass(frozen=True)
class Config:
    env_mode: EnvModeType
    selected_subsets: List[str] = field(
        default_factory=lambda: ["SUDOKU_4", "SUDOKU_5", "SUDOKU_6"]
    )

    # Derived fields (init=False)
    data_root: Path = field(init=False)
    base_dir: Path = field(init=False)
    taco_raw_dir: Path = field(init=False)
    taco_file_paths: List[Path] = field(init=False)
    normalized_sets_dir: Path = field(init=False)
    finetune_dir: Path = field(init=False)
    train_dir: Path = field(init=False)
    val_dir: Path = field(init=False)
    test_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        # Set data_root based on env_mode (defaults; override in factory if needed)
        data_root_map = {
            "local": Path("/mnt/shared"),
            "remote": Path.home(),
            "colab": Path("/content/drive/MyDrive"),
        }
        object.__setattr__(
            self, "data_root", data_root_map.get(self.env_mode, Path.cwd())
        )

        # Derive other paths
        object.__setattr__(self, "base_dir", self.data_root / "datasets/sen2venus")
        # object.__setattr__(self, "taco_raw_dir", self.base_dir / "TACO_raw_data")
        # object.__setattr__(
        #     self,
        #     "taco_file_paths",
        #     [self.taco_raw_dir / f"{subset}.taco" for subset in self.selected_subsets],
        # )
        object.__setattr__(
            self, "normalized_sets_dir", self.base_dir / "normalized_sets"
        )
        object.__setattr__(self, "finetune_dir", self.base_dir / "finetune")
        object.__setattr__(self, "train_dir", self.normalized_sets_dir / "train")
        object.__setattr__(self, "val_dir", self.normalized_sets_dir / "val")
        object.__setattr__(self, "test_dir", self.normalized_sets_dir / "test")

    def validate(self) -> None:
        """Validate config paths exist; raise errors otherwise."""
        missing_paths = []
        for attr in [
            "data_root",
            "base_dir",
            # "taco_raw_dir",
            "normalized_sets_dir",
            "finetune_dir",
            "train_dir",
            "val_dir",
            "test_dir",
        ]:
            path: Path = getattr(self, attr)
            if not path.exists():
                missing_paths.append(str(path))
        if missing_paths:
            raise ValueError(f"Missing paths: {', '.join(missing_paths)}")
        # for file_path in self.taco_file_paths:
        #     if not file_path.exists():
        #         missing_paths.append(str(file_path))
        if missing_paths:
            raise ValueError(f"Missing taco files: {', '.join(missing_paths)}")


def setup_environment(env_mode: EnvModeType) -> None:
    """Perform environment-specific setup (side effects isolated here)."""
    if env_mode.startswith("colab"):
        try:
            import super_image  # noqa: F401
            # import rasterio  # noqa: F401
            # import tacoreader  # noqa: F401
        except ImportError:
            print("Installing external packages...")
            try:
                subprocess.run(
                    [
                        "pip",
                        "install",
                        "--quiet",
                        "super-image",
                        # "rasterio",
                        # "tacoreader",
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install package: {e}")

        try:
            from google.colab import drive

            drive.mount("/content/drive", force_remount=True)
        except ImportError:
            raise RuntimeError("Google Colab module not found. Are you in Colab?")
        except Exception as e:
            raise RuntimeError(f"Failed to mount Google Drive: {e}")

        # Optional: Copy data to local /content for faster I/O in Colab
        if env_mode.endswith("vm"):
            colab_vm_dir = Path("/content/taco_normalized")
            if not colab_vm_dir.exists():
                print(
                    "Copying normalized data to local Colab storage for performance..."
                )
                shutil.copytree(
                    Path("/content/drive/MyDrive/datasets/sen2venus/normalized_sets"),
                    colab_vm_dir,
                )
                print("Copy complete.")
            # Avoid os.chdir; let users handle working dir if needed

    elif env_mode == "remote":
        print("Remote environment detected. No specific setup needed.")

    elif env_mode == "local":
        print("Local environment detected. Ensuring dependencies...")


def create_config(env_mode: EnvModeType | None = None) -> Config:
    """Factory to create and setup config based on detected environment."""
    if env_mode is None:
        if "google.colab" in sys.modules:
            env_mode = "colab"
        elif "REMOTE_ENV_VAR" in os.environ:
            env_mode = "remote"
        else:
            env_mode = "local"

    setup_environment(env_mode)
    config = Config(env_mode=env_mode)
    config.validate()
    return config


if __name__ == "__main__":
    config = create_config()
    print(f"data root is {config.data_root}")
