import json

notebook_path = "notebooks/01_eda.ipynb"

# The direct download links are flaky/HTML walls.
# Robust solution: git clone the repo into a temp dir and move the file.
new_download_code = [
    '    DATA_DIR = "../data/raw"\n',
    '    os.makedirs(DATA_DIR, exist_ok=True)\n',
    '\n',
    '    def download_data(output_dir):\n',
    '        if os.path.exists(os.path.join(output_dir, "train_FD001.txt")):\n',
    '            print("Data already exists. Skipping download.")\n',
    '            return\n',
    '        \n',
    '        print("Downloading data via git clone...")\n',
    '        # Clone into a temp directory\n',
    '        temp_dir = "temp_cmapss_repo"\n',
    '        repo_url = "https://github.com/shining0611armor/Predicting-of-Turbofan-Engine-Degradation-Using-the-NASA-C-MAPSS-Dataset.git"\n',
    '        \n',
    '        # Use system commands for git\n',
    '        if os.path.exists(temp_dir):\n',
    '            import shutil\n',
    '            shutil.rmtree(temp_dir)\n',
    '            \n',
    '        os.system(f"git clone {repo_url} {temp_dir}")\n',
    '        \n',
    '        # Move the zip file\n',
    '        zip_path = os.path.join(temp_dir, "CMAPSSData.zip")\n',
    '        if os.path.exists(zip_path):\n',
    '            with zipfile.ZipFile(zip_path, "r") as z:\n',
    '                z.extractall(output_dir)\n',
    '            print("Download and extraction complete.")\n',
    '        else:\n',
    '            print("Error: CMAPSSData.zip not found in cloned repo.")\n',
    '            \n',
    '        # Cleanup\n',
    '        import shutil\n',
    '        if os.path.exists(temp_dir):\n',
    '            shutil.rmtree(temp_dir)\n',
    '\n',
    '    download_data(DATA_DIR)'
]

with open(notebook_path, "r") as f:
    notebook = json.load(f)

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        if "def download_data" in source_str:
            cell["source"] = new_download_code
            break

with open(notebook_path, "w") as f:
    json.dump(notebook, f, indent=1)
