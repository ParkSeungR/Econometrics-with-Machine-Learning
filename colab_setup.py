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
    b = path.read_bytes()
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        # �����쿡�� cp949�� ����� ��� �ڵ� ��ȯ
        text = b.decode("cp949")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # -r, --find-links ���� ���⼱ ����(�ܼ� ��� ����)
        if line.startswith("-"):
            continue
        lines.append(line)
    return lines

def _packages_to_install(req_lines: list[str],
                         protected: set[str],
                         mode: str = "missing_only") -> list[str]:
    """
    mode:
      - 'missing_only' : ���ų�(�̼�ġ) ����� ���� �䱸�� �������� ���� ���� ��ġ
      - 'strict'       : �׻� ��ġ(pip���� �ñ�)  �� �ʿ�� �ɼ����� ���
    """
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet
        import importlib.metadata as ilmd
    except Exception:
        # packaging�� ���ٸ� �켱 ��ġ
        _run(["pip", "install", "-q", "packaging"], quiet=False)
        from packaging.requirements import Requirement
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet
        import importlib.metadata as ilmd

    to_install = []
    for line in req_lines:
        # URL(��: git+https://...) �Ǵ� ���� ��� �䱸������ �״�� ��ġ ť�� ����
        if any(line.startswith(p) for p in ("git+", "http://", "https://", "file:")):
            to_install.append(line)
            continue

        try:
            req = Requirement(line)
        except Exception:
            # �Ľ� �Ұ��ϸ� pip���� �״�� �ñ�
            to_install.append(line)
            continue

        name = req.name.replace("_", "-").lower()
        if name in {p.lower() for p in protected}:
            # Colab �⺻ ��Ű���� ��ȣ(��ŵ)
            print(f"[INFO] skip protected package: {name}")
            continue

        installed_ver = None
        try:
            installed_ver = ilmd.version(name)
        except ilmd.PackageNotFoundError:
            installed_ver = None

        if installed_ver is None:
            # �̼�ġ �� ��ġ ���
            to_install.append(line)
            continue

        if mode == "strict":
            to_install.append(line)
            continue

        # missing_only ���: ���� ������ ������ ���� ���� Ȯ��
        if req.specifier:  # e.g., >=, ==, < ��
            if not req.specifier.contains(installed_ver, prereleases=True):
                to_install.append(line)
        else:
            # ���� ������ ���� �̹� ��ġ�Ǿ� ������ ��ŵ
            pass

    return to_install

def setup_project(
    repo_url="https://github.com/ParkSeungR/Econometrics-with-Machine-Learning.git",
    repo_name="Econometrics-with-Machine-Learning",
    branch="main",
    data_dir="Data",
    func_pkg="Functions",
    requirements_file="requirements.txt",
    install_requirements=True,
    install_mode="missing_only",  # 'missing_only' �Ǵ� 'strict'
    protected_packages=None        # �⺻ None �� Colab �⺻ ��Ű�� ��Ʈ ���
):
    """
    Google Colab ȯ�� ����:
      - ����� ����ȭ
      - (�ɼ�) requirements.txt �о �ʿ��� �͸� ��ġ
      - ����� ��Ʈ�� sys.path�� �߰� �� `from Functions import ...` ����
    """
    repo_path = Path("/content") / repo_name

    print("[STEP] Sync repository...")
    if not repo_path.exists():
        _run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(repo_path)], quiet=False)
    else:
        _run(["git", "-C", str(repo_path), "fetch", "origin", branch, "--depth", "1"], quiet=True)
        _run(["git", "-C", str(repo_path), "reset", "--hard", f"origin/{branch}"], quiet=True)

    os.chdir(repo_path)
    repo_root = str(repo_path.resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Functions ��Ű�� ����Ʈ ����(����� ��Ʈ�� path�� �߰������Ƿ� OK)
    # �ʿ��ϸ� ���� ��� ��ε� �߰� ����:
    # func_dir_abs = str((repo_path / func_pkg).resolve())
    # if func_dir_abs not in sys.path:
    #     sys.path.append(func_dir_abs)

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
                # �� ���� ��ġ
                _run(["pip", "install", "-q"] + targets, quiet=False)
            else:
                print("[INFO] all requirements already satisfied (or protected).")
        else:
            print(f"[INFO] {requirements_file} not found. Skip.")

    print("[READY] Project is ready.")
    print(f"Repo root     : {repo_root}")
    print(f"Data directory: {repo_path / data_dir}")
    print(f"Functions pkg : {func_pkg} (importable)")

    # ���� ��� �Լ�
    def p(*rel):
        return str(Path(repo_root).joinpath(*rel))
    globals()["p"] = p

if __name__ == "__main__":
    setup_project()