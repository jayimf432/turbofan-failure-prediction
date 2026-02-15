import json

notebook_path = "notebooks/01_eda.ipynb"

# Simplify the notebook logic to just use the existing data if present.
new_download_code = [
    '    DATA_DIR = "../data/raw"\n',
    '    os.makedirs(DATA_DIR, exist_ok=True)\n',
    '\n',
    '    def download_data(output_dir):\n',
    '        # Just check if files exist, if not, try the git clone fallback\n',
    '        if os.path.exists(os.path.join(output_dir, "train_FD001.txt")):\n',
    '            print("Data found in " + output_dir)\n',
    '            return\n',
    '        \n',
    '        print("Data source not found. Please ensure data is in ../data/raw")\n',
    '        print("You can download it manually or use the provided script.")\n',
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
