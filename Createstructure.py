import os

# Base directory
base_dir = "AEW"

# Models and sample experiments (update as needed)
models = ["baseline", "newmodel12", "newmodel13"]
experiments = [
    "exp_baseline_300_lr0.001_20250922",
    "exp_newmodel12_300_focal_20250922",
    "exp_newmodel13_500_aug1_20250921"
]

# Folder structure
folders = [
    "notebooks/experiments",
    "notebooks/exploration",
    "src",
    "docs",
    "scripts",
    "tests"
]

# Create base folders
for folder in folders:
    path = os.path.join(base_dir, folder)
    os.makedirs(path, exist_ok=True)

# Create results structure
for model in models:
    model_dir = os.path.join(base_dir, "results", model)
    os.makedirs(model_dir, exist_ok=True)
    for exp in experiments:
        if exp.startswith(model):
            exp_dir = os.path.join(model_dir, exp)
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "saliency_maps"), exist_ok=True)

# Create placeholder files
placeholder_files = [
    os.path.join(base_dir, "README.md"),
    os.path.join(base_dir, ".gitignore"),
    os.path.join(base_dir, "requirements.txt"),
    os.path.join(base_dir, "src", "__init__.py"),
    os.path.join(base_dir, "tests", "__init__.py"),
]

for file in placeholder_files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write("# Placeholder\n")

print(f"AEW repo structure created under '{base_dir}'")
