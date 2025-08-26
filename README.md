## How to Run the Inference Script
> prerequisites: Anaconda or Miniconda for local setup
### **Step 1:** Download the pre-trained models
Download the pre-trained models from the top-most `models` directory in the repository.
Extract them in the directory of your choice. You will use these paths once you reach step 3.

### **Step 2:**  Activate conda Environment
Navigate to `src/super-res` and open a terminal window or command prompt.

**1. Create**
```
conda env create -f environment.yml
```
**2. Activate**
```
conda activate super_res_inf_env
```
### **Step 3:** Configuring .env 

Specify the exact paths to the extracted checkpoint files using a `.env` file.

**Steps:**

1.  **Create a `.env` file:** In the  `src/super-res` directory, make a copy of the `.env.example` file and rename it to `.env`.

2.  **Edit the `.env` file:** Open the new `.env` file and set the `CKPT_PATH_EDSR_16` and `CKPT_PATH_EDSR_8` variables to the absolute paths of your saved `.pt` checkpoint files.

### **Step 4:** 

1.  Make sure `inference.py` , `config.py` and `.env` are in the same directory.
2.  **Run from your terminal:** Open a terminal or command prompt in that directory.

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


