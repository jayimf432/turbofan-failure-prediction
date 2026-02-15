import os
import shutil

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data/raw")
os.makedirs(data_dir, exist_ok=True)

print(f"Target data directory: {data_dir}")

# URL for the repo
repo_url = "https://github.com/mapr-demos/predictive-maintenance.git"
temp_dir = "temp_mapr_repo_manual"

# Clean up temp dir if exists
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

print("Cloning repository...")
ret = os.system(f"git clone {repo_url} {temp_dir}")

if ret != 0:
    print("Git clone failed.")
    exit(1)

# Copy the files
source_dir = os.path.join(temp_dir, "notebooks/jupyter/Dataset/CMAPSSData")
if os.path.exists(source_dir):
    print(f"Found dataset at {source_dir}. Copying...")
    for filename in os.listdir(source_dir):
        if filename.endswith(".txt"):
            shutil.copy(os.path.join(source_dir, filename), data_dir)
    print("Copy complete.")
    
    # Verify extraction
    files = os.listdir(data_dir)
    print(f"Files in {data_dir}: {files}")

else:
    print("Error: Dataset folder not found in cloned repo.")

# Cleanup
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
