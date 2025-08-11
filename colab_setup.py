# -*- coding: utf-8 -*-
import os
import subprocess
from pathlib import Path

def setup_project():
    repo_url = "https://github.com/ParkSeungR/Econometrics-with-Machine-Learning.git"
    repo_name = "Econometrics-with-Machine-Learning"
    data_dir = "Data"
    func_dir = "Functions"

    print("[STEP] Checking repository...")
    if not Path(repo_name).exists():
        print("[INFO] Cloning repository...")
        subprocess.run(["git", "clone", repo_url])
    else:
        print("[INFO] Pulling latest changes...")
        subprocess.run(["git", "-C", repo_name, "pull"])

    # Change working directory to repo
    os.chdir(repo_name)

    # Install requirements.txt if exists
    req_file = Path("requirements.txt")
    if req_file.exists():
        print("[STEP] Installing requirements...")
        subprocess.run(["pip", "install", "-r", str(req_file)])
    else:
        print("[INFO] requirements.txt not found. Skipping.")

    # Check Data directory
    if Path(data_dir).exists():
        print(f"[OK] {data_dir}/ ready")
    else:
        print(f"[WARN] {data_dir}/ directory not found")

    # Check Functions directory
    if Path(func_dir).exists():
        print(f"[OK] {func_dir}/ ready")
        # Add Functions to sys.path
        import sys
        sys.path.append(str(Path(func_dir).resolve()))
        print(f"[INFO] {func_dir}/ added to Python path")
    else:
        print(f"[WARN] {func_dir}/ directory not found")

    print("[READY] Project environment is ready.")

if __name__ == "__main__":
    setup_project()