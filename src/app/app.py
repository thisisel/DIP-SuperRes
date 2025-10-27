import streamlit as st
from pathlib import Path

import config
import data
import processing

# --- Page Configuration ---
st.set_page_config(page_title="EDSR Super-Resolution Demo", layout="wide")

# --- App State Management ---
# Use session_state to store the config and results to avoid reloading on every interaction.
if "app_config" not in st.session_state:
    st.session_state.app_config = None
if "result" not in st.session_state:
    st.session_state.result = None

# --- UI Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")

    # The key for local development: a toggle for mock mode
    use_mock_mode = st.toggle(
        "Run in Mock Mode (CPU-only)",
        value=True,
        help="Simulates the model on your local CPU for fast UI development. Uncheck this on a GPU server for real inference.",
    )
    env_mode = (
        "local-mock" if use_mock_mode else "remote"
    )  # Or 'local' based on your real env

    # Initialize the config based on the selected mode
    try:
        st.session_state.app_config = config.create_config(env_mode=env_mode)
    except Exception as e:
        st.error(f"Error initializing configuration: {e}")
        st.stop()

    st.header("1. Select Input Image")

    # Get pre-loaded example images
    example_paths = data.get_example_image_paths(st.session_state.app_config)
    example_options = {path.name: path for path in example_paths}

    selected_example = st.selectbox(
        "Choose an example:",
        options=list(example_options.keys()),
        index=0,
        help="Select a pre-loaded image from the test set to run in 'Evaluation Mode'.",
    )

    st.write("--- or ---")

    uploaded_file = st.file_uploader(
        "Upload your own image:", type=["png", "jpg", "jpeg"]
    )

    st.header("2. Choose Model")
    model_arch_map = {
        "EDSR 16-Block (High Quality)": "EDSR_16",
        "EDSR 8-Block (Fast)": "EDSR_8",
    }
    selected_model_display = st.radio(
        "Select architecture:", list(model_arch_map.keys())
    )
    model_arch = model_arch_map[selected_model_display]

    # --- Action Button ---
    if st.button("🚀 Generate Super-Resolution"):
        # Determine the input source
        if uploaded_file is not None:
            # User upload takes precedence
            input_lr_path = uploaded_file
            input_hr_path = None  # No ground truth for uploads
        else:
            # Fallback to selected example
            input_lr_path = example_options[selected_example]
            # Try to find the corresponding HR image
            hr_path = Path(str(input_lr_path).replace("/lr/", "/hr/"))
            hr_path = Path(str(hr_path).replace("lr", "hr"))
            # hr_path = Path(str(input_lr_path).replace("/lr/", "/hr/").replace("lr", "hr"))
            input_hr_path = hr_path if hr_path.exists() else None

        with st.spinner(f"Running {selected_model_display}, please wait..."):
            try:
                st.session_state.result = processing.process_image_for_app(
                    config=st.session_state.app_config,
                    model_arch=model_arch,
                    # lr_path=str(
                    #     input_lr_path
                    # ),  # Ensure path is a string for the function
                    lr_path=input_lr_path,
                    hr_path=str(input_hr_path) if input_hr_path else None,
                )
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.session_state.result = None

# --- Main Content Area ---
st.title("🖼️ EDSR Super-Resolution Results")

if st.session_state.result is None:
    st.info(
        "Configure your settings in the sidebar and click 'Generate' to see the results."
    )
else:
    res = st.session_state.result
    st.subheader(f"Displaying results for: `{res.input_source_name}`")

    # --- Conditional Display Logic ---
    if res.metrics:
        # "Evaluation Mode" with 3 columns and metrics
        st.success("Running in **Evaluation Mode** (Ground Truth is available).")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(
                res.images["bicubic"],
                caption=f"Bicubic | PSNR: {res.metrics['bicubic']['psnr']:.2f}, SSIM: {res.metrics['bicubic']['ssim']:.4f}",
                width='stretch',
            )
        with col2:
            st.image(
                res.images["sr"],
                caption=f"Super-Resolved | PSNR: {res.metrics['sr']['psnr']:.2f}, SSIM: {res.metrics['sr']['ssim']:.4f}",
                width='stretch',
            )
        with col3:
            st.image(
                res.images["hr"],
                caption="Ground Truth (High-Resolution)",
                width='stretch',
            )

    else:
        # "Inference Mode" with 2 columns
        st.warning(
            "Running in **Inference Mode** (No Ground Truth available, metrics cannot be calculated)."
        )

        col1, col2 = st.columns(2)
        with col1:
            # For user uploads, the "lr" image is their original
            st.image(
                res.images["lr"], caption="Original User Input", width='stretch'
            )
        with col2:
            st.image(
                res.images["sr"], caption="Super-Resolved Output", width='stretch'
            )
