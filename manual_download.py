import os
import zipfile
import shutil

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data/raw")
os.makedirs(data_dir, exist_ok=True)

print(f"Target data directory: {data_dir}")

# URL for the repo
repo_url = "https://github.com/shining0611armor/Predicting-of-Turbofan-Engine-Degradation-Using-the-NASA-C-MAPSS-Dataset.git"
temp_dir = "temp_cmapss_repo"

# Clean up temp dir if exists
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

print("Cloning repository...")
ret = os.system(f"git clone {repo_url} {temp_dir}")

if ret != 0:
    print("Git clone failed.")
    exit(1)

zip_path = os.path.join(temp_dir, "CMAPSSData.zip")
if os.path.exists(zip_path):
    print(f"Found zip file at {zip_path}. Extracting...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print("Extraction complete.")
        
        # Verify extraction
        files = os.listdir(data_dir)
        print(f"Files in {data_dir}: {files}")
        
    except zipfile.BadZipFile:
        print("Error: The file is not a valid zip file.")
else:
    print("Error: CMAPSSData.zip not found in the cloned repository.")

# Cleanup
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
