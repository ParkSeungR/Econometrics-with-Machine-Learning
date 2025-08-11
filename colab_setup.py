# colab_setup.py
import os
import sys
import subprocess
from pathlib import Path

def _run(cmd, check=True, quiet=False):
    if quiet:
        subprocess.run(cmd, check=check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, check=check)

def setup_project(
    repo_url="https://github.com/ParkSeungR/Econometrics-with-Machine-Learning.git",
    repo_name="Econometrics-with-Machine-Learning",
    branch="main",
    data_dir="Data",
    func_pkg="Functions",
    requirements_file="requirements.txt",
    install_requirements=True
):
    """Prepare Google Colab environment for this project."""
    repo_path = Path("/content") / repo_name

    print("[STEP] Sync repository...")
    if not repo_path.exists():
        _run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(repo_path)], quiet=False)
    else:
        # get latest
        _run(["git", "-C", str(repo_path), "fetch", "origin", branch, "--depth", "1"], quiet=True)
        _run(["git", "-C", str(repo_path), "reset", "--hard", f"origin/{branch}"], quiet=True)

    # change working directory to repo root
    os.chdir(repo_path)

    # ensure repo root on sys.path so `import Functions` works
    repo_root = str(repo_path.resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # optional: install requirements
    if install_requirements:
        req = Path(requirements_file)
        if req.exists():
            print("[STEP] Installing requirements...")
            _run(["pip", "install", "-q", "-r", str(req)], quiet=False)
        else:
            print(f"[INFO] {requirements_file} not found. Skip.")

    # quick checks
    print(f"[READY] Project is ready.")
    print(f"Repo root     : {repo_root}")
    print(f"Data directory: {repo_path / data_dir}")
    print(f"Functions pkg : {func_pkg} (importable)")

    # return some handy helpers (optional)
    def p(*rel):
        return str(Path(repo_root).joinpath(*rel))
    globals()["p"] = p  # so user can call p("Data/file.csv")

if __name__ == "__main__":
    setup_project()