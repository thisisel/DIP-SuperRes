# Super-Resolution on Sentinel-2 Satellite Imagery

This project implements and evaluates a deep learning-based super-resolution model for enhancing the spatial resolution of Sentinel-2 satellite imagery. The core of the project is a fine-tuned EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution) model, which was trained on the SEN2VENµS dataset.

This repository contains the full training and inference pipeline, along with a Streamlit-based web application for interactive demonstrations.

## Features

- **Two Model Architectures:**
  - **EDSR 16-Block (High Quality):** The primary fine-tuned model, delivering the best performance.
  - **EDSR 8-Block (Fast):** A lighter variant for faster inference, trained from scratch.
- **Robust Inference Pipeline:** Handles various image sources (local files, URLs, uploaded data) and correctly preprocesses them to match the model's training conditions.
- **Interactive UI:** Select example images or upload your own to see a side-by-side comparison of Bicubic, Super-Resolved, and Ground Truth (if available) results.

## How to Run the Inference
> **prerequisites:** 
> * `git` for cloning the repository.
> * Python 3.9+
> * Python package manger: pip or conda
> * For local GPU support: An NVIDIA GPU with CUDA and cuDNN installed.
### **Step 1:** Clone the repository

```bash
git clone https://github.com/thisisel/DIP-SuperRes.git
cd DIP-SuperRes
```
Pre-trained models are located in the top-most `models` directory
.Extract them in the directory of your choice. 
#### **Project Structure**
```
├── environment.yml --> packages for training and evaluation
├── figures
│   ├── evaluation-plot.jpeg
│   └── tables
│       ├── 16_block_edsr_train_val_performance.csv
│       └── 8_block_edsr_train_val_performance.csv
├── models 
│   ├── edsr_base_16_block.zip  --> unzip and make note of abs path
│   └── edsr_base_8_block.zip  --> unzip and make note of abs path
├── notebooks
│   ├── data.ipynb
│   ├── eval.ipynb
│   ├── finetune.ipynb
│   ├── shared_config.py
│   └── train.ipynb
├── src
│   ├── requirements.txt  --> packages for inference
│   ├── app  --> app root directory
│   │   ├── app.py  --> Main Streamlit application script
│   │   ├── assets
│   │   │   ├── examples
│   │   │   │   ├── hr
│   │   │   │   └── lr
│   │   │   └── examples_npy
│   │   │       ├── hr
│   │   │       └── lr
│   │   ├── config.py  --> App specific Configuration file
│   │   ├── data.py  --> Scripts for data handling and preprocessing
│   │   ├── models.py
│   │   ├── processing.py
│   └── super-res --> inference script root dir
│       ├── config.py
│       ├── environment.yml --> packages for inference
│       └── inference.py
│        
```

### **Step 2:**  Setup Virtual Environment
```bash
cd src
```

**2.1. Create**
```bash
python -m venv venv
```
or
```bash
conda env create -f environment.yml
```
**2.2. Activate**
```bash
# On macOS and Linux:
source venv/bin/activate

# On Windows (Command Prompt or PowerShell):
 venv\Scripts\activate

 pip install -r requirements.txt
```
or
```bash
conda activate super_res_inf_env
```
### **Option A: Inference App**
#### **A.2:**
Make sure streamlit is installed
```bash
pip install streamlit
```
#### **A.2:**

Specify the exact paths to the extracted checkpoints files using a `.env` file.

1.  **Edit the `.env` file:** Open  `.env` file and set the `CKPT_PATH_EDSR_16` and `CKPT_PATH_EDSR_8` variables to the absolute paths of your saved `.pt` checkpoint files.

Navigate to app directory and launch the app from your terminal.
```bash
cd app
streamlit run app.py
```
The application should now be running and accessible in your web browser at `http://localhost:8501`.

### **Option B: Inference Script**
#### **B.1:**

Specify the exact paths to the extracted checkpoints files using a `.env` file.


1.  **Create a `.env` file:** In the  `src/super-res` directory, make a copy of the `.env.example` file and rename it to `.env`.

2.  **Edit the `.env` file:** Open the new `.env` file and set the `CKPT_PATH_EDSR_16` and `CKPT_PATH_EDSR_8` variables to the absolute paths of your saved `.pt` checkpoint files.

#### **B.2:** 

1.  Make sure `inference.py` , `config.py` and `.env` are in the same directory.
2.  **Run from your terminal:** Open a terminal or command prompt in `src/super-res`.

| Argument       | Type      | Description                                                           | valid inputs               |
|:---------------|:----------|:----------------------------------------------------------------------|:---------------------------|
| model-arch     | mandatory | The model architecture to use for inference&nbsp;                     | `EDSR_16`,&nbsp; `EDSR_8`  |
| input-path     | mandatory | Path or URL to the LR input image                                     |                            |
| output-dir     | optional  | Directory to save the output image. Defaults to the current directory |                            |
| env-mode<br> | optional  | Override automatic environment detection                              | `remote`, `local`, `colab` |  

**Example 1: Using the 16-block model on a local file**

```bash
python inference.py --model-arch EDSR_16 --input-path /path/to/your/image.png
```

**Example 2: Using the 8-block model and saving to a specific directory**

```bash
python inference.py --model-arch EDSR_8 --input-path /path/to/another/image.jpg --output-dir /path/to/save/results
```

**Example 3: Using a URL as input**

```bash
python inference.py --model-arch EDSR_16 --input-path "https://example.com/some_low_res_image.png"
```

**Example 4: Running in Colab and overriding environment detection**

If you were running this from a terminal inside a Colab instance, you might use:

```bash
python inference.py --model-arch EDSR_16 --input-path "/content/my_test_image.png" --env-mode colab
```


## Training Pipeline
The model was trained using a custom, resumable PyTorch trainer detailed in the `notebooks/Training.ipynb` notebook. The pipeline is designed to be resilient to disconnections and provides detailed logging for experimental reproducibility.

## Acknowledgements
- This project is based on the paper: ["Enhanced Deep Residual Networks for Single Image Super-Resolution"](https://arxiv.org/abs/1707.02921) by Bee Lim et al.
- The `super-image` library was used as a foundational tool for the EDSR model implementation.
- The SEN2VENµS dataset was provided by the [TACO Foundation](https://www.tacofoundation.org/).


