# colab_setup.py
import os
import sys
import subprocess
from pathlib import Path

def _run(cmd, check=True, quiet=False):
    subprocess.run(
        cmd, check=check,
        stdout=(subprocess.DEVNULL if quiet else None),
        stderr=(subprocess.DEVNULL if quiet else None)
    )

def _read_requirements_utf8(path: Path) -> list[str]:
    """Read requirements file as UTF-8; fallback to cp949 if needed. Returns normalized lines (no comments/flags)."""
    b = path.read_bytes()
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        text = b.decode("cp949")
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("-"):  # skip flags like -r, --find-links, etc.
            continue
        out.append(s)
    return out

def _packages_to_install(req_lines: list[str],
                         protected: set[str],
                         mode: str = "missing_only") -> list[str]:
    """
    Decide which packages to install.

    mode:
      - 'missing_only': install only if not installed or version does not satisfy spec
      - 'strict'     : always pass to pip (let pip resolve)
    """
    try:
        from packaging.requirements import Requirement
        import importlib.metadata as ilmd
    except Exception:
        _run(["pip", "install", "-q", "packaging"], quiet=False)
        from packaging.requirements import Requirement
        import importlib.metadata as ilmd

    to_install = []
    protected_lc = {p.lower() for p in protected}

    for line in req_lines:
        # Allow VCS/URL/direct paths as-is
        if line.startswith(("git+", "http://", "https://", "file:")):
            to_install.append(line)
            continue

        try:
            req = Requirement(line)
        except Exception:
            # if parsing fails, let pip try
            to_install.append(line)
            continue

        name_lc = req.name.replace("_", "-").lower()
        if name_lc in protected_lc:
            print(f"[INFO] skip protected package: {req.name}")
            continue

        installed_ver = None
        try:
            installed_ver = ilmd.version(name_lc)
        except ilmd.PackageNotFoundError:
            installed_ver = None

        if installed_ver is None:
            to_install.append(line)
            continue

        if mode == "strict":
            to_install.append(line)
            continue

        # missing_only: install only if spec exists and current version does not satisfy
        if req.specifier and not req.specifier.contains(installed_ver, prereleases=True):
            to_install.append(line)

    return to_install

def setup_project(
    repo_url="https://github.com/ParkSeungR/Econometrics-with-Machine-Learning.git",
    repo_name="Econometrics-with-Machine-Learning",
    branch="main",
    data_dir="Data",
    func_pkg="Functions",
    requirements_file="requirements.txt",
    install_requirements=True,
    install_mode="missing_only",      # 'missing_only' or 'strict'
    protected_packages=None            # default None -> use Colab core set
):
    """Prepare Google Colab environment for this project."""
    repo_path = Path("/content") / repo_name

    print("[STEP] Sync repository...")
    if not repo_path.exists():
        _run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(repo_path)], quiet=False)
    else:
        _run(["git", "-C", str(repo_path), "fetch", "origin", branch, "--depth", "1"], quiet=True)
        _run(["git", "-C", str(repo_path), "reset", "--hard", f"origin/{branch}"], quiet=True)

    os.chdir(repo_path)
    repo_root = str(repo_path.resolve())

    # ensure repo root on sys.path so 'import Functions' works
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if install_requirements:
        req = Path(requirements_file)
        if req.exists():
            print("[STEP] Installing requirements (filtered)...")
            lines = _read_requirements_utf8(req)
            if protected_packages is None:
                protected_packages = {
                    "google-colab", "jupyter-server", "notebook",
                    "ipykernel", "jupyterlab", "jupyterlab-server"
                }
            targets = _packages_to_install(lines, protected_packages, mode=install_mode)
            if targets:
                print("[INFO] to install:", ", ".join(targets))
                _run(["pip", "install", "-q"] + targets, quiet=False)
            else:
                print("[INFO] all requirements already satisfied or protected.")
        else:
            print(f"[INFO] {requirements_file} not found. Skip.")

    print("[READY] Project is ready.")
    print(f"Repo root     : {repo_root}")
    print(f"Data directory: {repo_path / data_dir}")
    print(f"Functions pkg : {func_pkg} (importable)")

    # handy path helper
    def p(*rel):
        return str(Path(repo_root).joinpath(*rel))
    globals()["p"] = p

if __name__ == "__main__":
    setup_project()