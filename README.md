Our code is modified based on ESM2. For more information about https://huggingface.co/facebook/esm2_t33_650M_UR50D
# ACP-ESM2
## Step by step for training model	
### 1.1 Create and activate a new virtual environment
conda create -n ESM2 python=3.6 <br>
conda activate ESM2
### 1.2 Install the package and other requirements
python3 -m pip install -r requirements.txt
### 2. Download pre-trained ESM2
https://huggingface.co/facebook/esm2_t33_650M_UR50D
### 3. Predict command 
export MODEL_PATH=./model/model.pkl
python test.py
