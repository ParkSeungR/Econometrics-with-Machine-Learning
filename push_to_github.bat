@echo off
setlocal

:: ===== ����� ���� =====
set "GITHUB_USER=ParkSeungR"
set "REPO_NAME=Econometrics-with-Machine-Learning"
set "BRANCH=main"
:: =======================

echo.
echo =========================================================
echo   GitHub Push - %REPO_NAME% (%BRANCH%)
echo =========================================================

:: Git ��ġ Ȯ��
git --version >nul 2>&1 || (
  echo [ERROR] Git �̼�ġ. https://git-scm.com/download/win ��ġ �� ��õ�.
  pause & exit /b 1
)

:: ����� �ʱ�ȭ/���� ����
if not exist ".git" git init
set "REPO_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git"
git remote get-url origin >nul 2>&1 || git remote add origin "%REPO_URL%"
git checkout -q -B %BRANCH%

:: (�� ����) ���� ����� ���: ù Ǫ�� �� ���̵�/��ū �Է� �� �����
git config --global credential.helper manager

:: ���� �ֽ� �ݿ�
echo [STEP] pull --rebase
git fetch origin
git pull --rebase --autostash origin %BRANCH% || (
  echo [ERROR] pull �� �浹. ���� ������ git add �� git rebase --continue �� �ٽ� ����.
  pause & exit /b 1
)

:: ���� ���� ������¡
git add -A

:: ������ ������ Ŀ�� ��ŵ
git diff --cached --quiet
if %errorlevel%==0 (
  echo [INFO] Ŀ���� ���� ����.
) else (
  git commit -m "Auto-commit"
  if errorlevel 1 (
    echo [ERROR] Ŀ�� ����.
    pause & exit /b 1
  )
)

:: Ǫ��
echo [STEP] push
git rev-parse --abbrev-ref --symbolic-full-name @{u} >nul 2>&1
if errorlevel 1 (
  git push -u origin %BRANCH% || ( echo [ERROR] Ǫ�� ����. ����/��ū Ȯ��. & pause & exit /b 1 )
) else (
  git push || ( echo [ERROR] Ǫ�� ����. ����/��ū Ȯ��. & pause & exit /b 1 )
)

echo [OK] ����ȭ �Ϸ� �� %REPO_URL%
pause