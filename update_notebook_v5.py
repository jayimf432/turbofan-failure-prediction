import json

notebook_path = "notebooks/01_eda.ipynb"

# Robust solution: git clone the mapr-demos repo and copy the files.
new_download_code = [
    '    DATA_DIR = "../data/raw"\n',
    '    os.makedirs(DATA_DIR, exist_ok=True)\n',
    '\n',
    '    def download_data(output_dir):\n',
    '        if os.path.exists(os.path.join(output_dir, "train_FD001.txt")):\n',
    '            print("Data already exists. Skipping download.")\n',
    '            return\n',
    '        \n',
    '        print("Downloading data via git clone from mapr-demos...")\n',
    '        # Clone into a temp directory\n',
    '        temp_dir = "temp_mapr_repo"\n',
    '        repo_url = "https://github.com/mapr-demos/predictive-maintenance.git"\n',
    '        \n',
    '        if os.path.exists(temp_dir):\n',
    '            import shutil\n',
    '            shutil.rmtree(temp_dir)\n',
    '            \n',
    '        os.system(f"git clone {repo_url} {temp_dir}")\n',
    '        \n',
    '        # Copy the files\n',
    '        source_dir = os.path.join(temp_dir, "notebooks/jupyter/Dataset/CMAPSSData")\n',
    '        if os.path.exists(source_dir):\n',
    '            import shutil\n',
    '            for filename in os.listdir(source_dir):\n',
    '                if filename.endswith(".txt"):\n',
    '                    shutil.copy(os.path.join(source_dir, filename), output_dir)\n',
    '            print("Download and extraction complete.")\n',
    '        else:\n',
    '            print("Error: Dataset folder not found in cloned repo.")\n',
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
