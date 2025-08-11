@echo off
setlocal

:: ===== ����� ���� =====
set "GITHUB_USER=ParkSeungR"
set "REPO_NAME=Econometrics-with-Machine-Learning"
set "BRANCH=main"
set "STRICT_SYNC=1"   :: 1=������ ���ð� ������ �����ϰ�(���ÿ� ���� ������ ���ݿ��� ����)
:: =======================

echo.
echo =========================================================
echo   GitHub Push - %REPO_NAME% (%BRANCH%)
echo =========================================================

:: Git Ȯ��
git --version >nul 2>&1 || ( echo [ERROR] Git �̼�ġ. https://git-scm.com/download/win ; pause & exit /b 1 )

:: ����� �ʱ�ȭ/����
if not exist ".git" git init
set "REPO_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git"
git remote get-url origin >nul 2>&1 || git remote add origin "%REPO_URL%"
git config --global credential.helper manager
git checkout -q -B %BRANCH%

:: ���� �ֽ� �ݿ�
echo [STEP] pull --rebase
git fetch origin
git pull --rebase --autostash origin %BRANCH% || (
  echo [ERROR] pull �浹. ���� ���� �� git add �� git rebase --continue ; pause & exit /b 1
)

:: === STRICT ���: �ε��� �ʱ�ȭ �� ���� ���� �������� ���� ===
if "%STRICT_SYNC%"=="1" (
  echo [STEP] strict sync: repo index reset to local snapshot
  git rm -r --cached . >nul 2>&1
)

:: ���� ������¡(���� ����)
git add -A

:: Ŀ��(���� ������ ��ŵ)
git diff --cached --quiet
if %errorlevel%==0 (
  echo [INFO] Ŀ���� ���� ����.
) else (
  for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set TODAY=%%a-%%b-%%c
  set NOW=%time: =0%
  set NOW=%NOW::=_%
  git commit -m "Auto-commit (strict sync): %TODAY% %NOW%" || ( echo [ERROR] Ŀ�� ���� ; pause & exit /b 1 )
)

:: Ǫ��
echo [STEP] push
git rev-parse --abbrev-ref --symbolic-full-name @{u} >nul 2>&1
if errorlevel 1 ( git push -u origin %BRANCH% ) else ( git push ) || ( echo [ERROR] Ǫ�� ����. ����/��ū Ȯ�� ; pause & exit /b 1 )

echo [OK] ����ȭ �Ϸ� �� %REPO_URL%
pause