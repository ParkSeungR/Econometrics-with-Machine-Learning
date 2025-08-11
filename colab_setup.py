def setup_project(repo_owner="ParkSeungR", repo_name="Econometrics-with-Machine-Learning",
                  branch="main", data_dir="Data", func_dir="Functions",
                  install_requirements=True, req_file="requirements.txt"):
    import os, sys, subprocess, pandas as pd

    repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
    local_root = f"/content/{repo_name}"

    def run(cmd, quiet=True):
        if isinstance(cmd, str):
            cmd = cmd.split()
        subprocess.run(cmd, check=True,
                       stdout=(subprocess.DEVNULL if quiet else None),
                       stderr=(subprocess.DEVNULL if quiet else None))

    # 1) 저장소 클론/업데이트
    if not os.path.exists(local_root):
        print(f"[clone] {repo_url} ({branch})")
        run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, local_root], quiet=False)
    else:
        print(f"[update] pulling latest from {branch}")
        run(["git", "-C", local_root, "fetch", "origin", branch, "--depth", "1"])
        run(["git", "-C", local_root, "reset", "--hard", f"origin/{branch}"])

    # 2) requirements 설치(옵션)
    if install_requirements:
        req_path = os.path.join(local_root, req_file)
        if os.path.exists(req_path):
            print(f"[install] requirements from {req_file}")
            run(["pip", "install", "-q", "-r", req_path], quiet=False)
        else:
            print(f"[skip] {req_file} not found")

    # 3) 경로 세팅
    os.chdir(local_root)
    func_path = os.path.join(local_root, func_dir)
    if func_path not in sys.path:
        sys.path.append(func_path)

    # 4) 편의 함수 등록
    def p(*rel):
        return os.path.join(local_root, *rel)

    def read_csv(rel_path, **kwargs):
        return pd.read_csv(p(rel_path), **kwargs)

    globals().update({"p": p, "read_csv": read_csv})
    print(f"[ready] {data_dir}/ 와 {func_dir}/ 사용 준비 완료")