## How to Run the Inference Script
> **prerequisites:** 
> * Python package manger: pip or conda
> * CUDA toolkit
### **Step 1:** Clone the repository
Download the pre-trained models from the top-most `models` directory in the repository.
Extract them in the directory of your choice. You will use these paths once you reach step 3.
```
git clone https://github.com/thisisel/DIP-SuperRes.git
cd src/super-res/
```
**Essential directories and files**
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
│   └── super-res --> inference root dir
│       ├── config.py
│       ├── environment.yml --> packages for inference
│       ├── inference.py
│       └── requirements.txt 
```

### **Step 2:**  Setup Virtual Environment
Make sure you are in `src/super-res`

**1. Create**
```
python -m venv venv
```
or
```
conda env create -f environment.yml
```
**2. Activate**
```
# On macOS and Linux:
source venv/bin/activate

# On Windows (Command Prompt or PowerShell):
 venv\Scripts\activate

 pip install -r requirements.txt
```
or
```
conda activate super_res_inf_env
```
### **Step 3:** Configuring .env 

Specify the exact paths to the extracted checkpoints files using a `.env` file.

**Steps:**

1.  **Create a `.env` file:** In the  `src/super-res` directory, make a copy of the `.env.example` file and rename it to `.env`.

2.  **Edit the `.env` file:** Open the new `.env` file and set the `CKPT_PATH_EDSR_16` and `CKPT_PATH_EDSR_8` variables to the absolute paths of your saved `.pt` checkpoint files.

### **Step 4:** 

1.  Make sure `inference.py` , `config.py` and `.env` are in the same directory.
2.  **Run from your terminal:** Open a terminal or command prompt in that directory.

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


