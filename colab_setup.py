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
        # 윈도우에서 cp949로 저장된 경우 자동 변환
        text = b.decode("cp949")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # -r, --find-links 등은 여기선 무시(단순 목록 가정)
        if line.startswith("-"):
            continue
        lines.append(line)
    return lines

def _packages_to_install(req_lines: list[str],
                         protected: set[str],
                         mode: str = "missing_only") -> list[str]:
    """
    mode:
      - 'missing_only' : 없거나(미설치) 명시적 버전 요구를 만족하지 못할 때만 설치
      - 'strict'       : 항상 설치(pip에게 맡김)  ← 필요시 옵션으로 사용
    """
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet
        import importlib.metadata as ilmd
    except Exception:
        # packaging이 없다면 우선 설치
        _run(["pip", "install", "-q", "packaging"], quiet=False)
        from packaging.requirements import Requirement
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet
        import importlib.metadata as ilmd

    to_install = []
    for line in req_lines:
        # URL(예: git+https://...) 또는 로컬 경로 요구사항은 그대로 설치 큐에 넣음
        if any(line.startswith(p) for p in ("git+", "http://", "https://", "file:")):
            to_install.append(line)
            continue

        try:
            req = Requirement(line)
        except Exception:
            # 파싱 불가하면 pip에게 그대로 맡김
            to_install.append(line)
            continue

        name = req.name.replace("_", "-").lower()
        if name in {p.lower() for p in protected}:
            # Colab 기본 패키지는 보호(스킵)
            print(f"[INFO] skip protected package: {name}")
            continue

        installed_ver = None
        try:
            installed_ver = ilmd.version(name)
        except ilmd.PackageNotFoundError:
            installed_ver = None

        if installed_ver is None:
            # 미설치 → 설치 대상
            to_install.append(line)
            continue

        if mode == "strict":
            to_install.append(line)
            continue

        # missing_only 모드: 버전 조건이 있으면 만족 여부 확인
        if req.specifier:  # e.g., >=, ==, < 등
            if not req.specifier.contains(installed_ver, prereleases=True):
                to_install.append(line)
        else:
            # 버전 조건이 없고 이미 설치되어 있으면 스킵
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
    install_mode="missing_only",  # 'missing_only' 또는 'strict'
    protected_packages=None        # 기본 None → Colab 기본 패키지 세트 사용
):
    """
    Google Colab 환경 세팅:
      - 저장소 동기화
      - (옵션) requirements.txt 읽어서 필요한 것만 설치
      - 저장소 루트를 sys.path에 추가 → `from Functions import ...` 가능
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

    # Functions 패키지 임포트 가능(저장소 루트를 path에 추가했으므로 OK)
    # 필요하면 개별 모듈 경로도 추가 가능:
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
                # 한 번에 설치
                _run(["pip", "install", "-q"] + targets, quiet=False)
            else:
                print("[INFO] all requirements already satisfied (or protected).")
        else:
            print(f"[INFO] {requirements_file} not found. Skip.")

    print("[READY] Project is ready.")
    print(f"Repo root     : {repo_root}")
    print(f"Data directory: {repo_path / data_dir}")
    print(f"Functions pkg : {func_pkg} (importable)")

    # 편의 경로 함수
    def p(*rel):
        return str(Path(repo_root).joinpath(*rel))
    globals()["p"] = p

if __name__ == "__main__":
    setup_project()