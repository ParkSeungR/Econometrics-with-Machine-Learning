# colab_setup.py
import os
import subprocess
import sys

def setup_project(
    data_dir="Data",
    func_dir="Functions",
    requirements_file="requirements.txt",
    repo_url="https://github.com/ParkSeungR/Econometrics-with-Machine-Learning.git"
):
    """Setup Google Colab environment for this project."""
    
    print("[STEP] Cloning repository (if not exists)...")
    if not os.path.exists("/content/Econometrics-with-Machine-Learning"):
        subprocess.run(["git", "clone", repo_url], check=True)
    else:
        print("[INFO] Repository already exists. Skipping clone.")

    # Set working directory
    os.chdir("/content/Econometrics-with-Machine-Learning")

    print("[STEP] Installing requirements...")
    if os.path.exists(requirements_file):
        subprocess.run(["pip", "install", "-r", requirements_file], check=True)
    else:
        print(f"[WARN] {requirements_file} not found. Skipping.")

    # Add Functions directory to Python path
    abs_func_dir = os.path.join(os.getcwd(), func_dir)
    if os.path.exists(abs_func_dir) and abs_func_dir not in sys.path:
        sys.path.append(abs_func_dir)

    print("[READY] Project setup complete.")
    print(f"Data directory: {os.path.join(os.getcwd(), data_dir)}")
    print(f"Functions directory: {abs_func_dir}")