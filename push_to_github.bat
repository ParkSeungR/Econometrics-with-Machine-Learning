@echo off
setlocal enabledelayedexpansion

:: ===== ����� ���� =====
set "GITHUB_USER=ParkSeungR"
set "REPO_NAME=Econometrics-with-Machine-Learning"
set "DEFAULT_BRANCH=main"
:: =======================

echo.
echo =========================================================
echo   GitHub Push Helper - %REPO_NAME%
echo =========================================================
echo.

:: 0) Git ��ġ Ȯ��
git --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Git �̼�ġ. https://git-scm.com/download/win ���� ��ġ �� ��õ�.
  pause
  exit /b 1
)

:: 1) git init
if not exist ".git" (
  echo [INFO] .git ���� ���� �� git init
  git init
)

:: 2) origin ����
set "REPO_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git"
git remote get-url origin >nul 2>&1
if errorlevel 1 (
  echo [INFO] origin �߰�: %REPO_URL%
  git remote add origin "%REPO_URL%"
) else (
  echo [INFO] origin ����
)

:: 3) �⺻ �귣ġ �̵�/����
git checkout -q -B %DEFAULT_BRANCH%

:: 4) .gitignore �غ�(������ ����)
set "GITIGNORE=.gitignore"
if not exist "%GITIGNORE%" (
  echo [INFO] .gitignore ����
  (
    echo # Python
    echo __pycache__/
    echo *.pyc
    echo .ipynb_checkpoints/
    echo .pytest_cache/
    echo .mypy_cache/
    echo
    echo # IDE/OS
    echo .vscode/
    echo .idea/
    echo .DS_Store
    echo Thumbs.db
    echo
    echo # Environments
    echo envs/
    echo venv/
    echo .venv/
    echo
    echo # Outputs / large or generated
    echo Output/
    echo **/tmp/
  )> "%GITIGNORE%"
) else (
  echo [INFO] .gitignore Ȯ��
  for %%R in (__pycache__/ *.pyc .ipynb_checkpoints/ .vscode/ .idea/ .DS_Store Thumbs.db envs/ venv/ .venv/ Output/ **/tmp/) do (
    findstr /x /c:"%%R" "%GITIGNORE%" >nul 2>&1 || echo %%R>>"%GITIGNORE%"
  )
)

:: 5) �� ���� ���� ����(.gitkeep)
for %%D in (Data Functions Figures Jupyter Notebooks Model Output Python x13as) do (
  if exist "%%D" (
    if not exist "%%D\.gitkeep" echo.> "%%D\.gitkeep"
  )
)

:: 6) �ʼ� ���� ���� Ȯ��
if not exist "colab_setup.py" (
  echo [WARN] colab_setup.py�� �����ϴ�. (��Ʈ�� ��ġ ����)
)

:: 7) ���� ���� �̸�����
echo.
echo [INFO] ���� ���� ���:
git status --short
echo.
set /p _go="�� ������ Ŀ��/Ǫ���ұ�? (Y/N): "
if /i not "%_go%"=="Y" (
  echo �����.
  pause
  exit /b 0
)

:: 8) ������¡/Ŀ��
git add -A
for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set TODAY=%%a-%%b-%%c
set NOW=%time: =0%
set NOW=%NOW::=_%
git commit -m "Auto-commit: %TODAY% %NOW%" 2>nul
if errorlevel 1 (
  echo [INFO] Ŀ���� ���� ���� ����.
) else (
  echo [OK] Ŀ�� �Ϸ�.
)

:: 9) Ǫ��
echo [INFO] GitHub�� Ǫ�� ��...
git push -u origin %DEFAULT_BRANCH%
if errorlevel 1 (
  echo.
  echo [ERROR] Ǫ�� ����.
  echo - GitHub �����/����/��ū Ȯ�� �ʿ�.
  pause
  exit /b 1
)

echo.
echo =========================================================
echo   �Ϸ�! GitHub: %REPO_URL%
echo =========================================================
pause