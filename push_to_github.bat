@echo off
setlocal enabledelayedexpansion

:: ===== 사용자 설정 =====
set "GITHUB_USER=ParkSeungR"
set "REPO_NAME=Econometrics-with-Machine-Learning"
set "DEFAULT_BRANCH=main"
:: =======================

echo.
echo =========================================================
echo   GitHub Push Helper - %REPO_NAME%
echo =========================================================
echo.

:: 0) Git 설치 확인
git --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Git 미설치. https://git-scm.com/download/win 에서 설치 후 재시도.
  pause
  exit /b 1
)

:: 1) git init
if not exist ".git" (
  echo [INFO] .git 폴더 없음 → git init
  git init
)

:: 2) origin 설정
set "REPO_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git"
git remote get-url origin >nul 2>&1
if errorlevel 1 (
  echo [INFO] origin 추가: %REPO_URL%
  git remote add origin "%REPO_URL%"
) else (
  echo [INFO] origin 존재
)

:: 3) 기본 브랜치 이동/생성
git checkout -q -B %DEFAULT_BRANCH%

:: 4) .gitignore 준비(없으면 생성)
set "GITIGNORE=.gitignore"
if not exist "%GITIGNORE%" (
  echo [INFO] .gitignore 생성
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
  echo [INFO] .gitignore 확인
  for %%R in (__pycache__/ *.pyc .ipynb_checkpoints/ .vscode/ .idea/ .DS_Store Thumbs.db envs/ venv/ .venv/ Output/ **/tmp/) do (
    findstr /x /c:"%%R" "%GITIGNORE%" >nul 2>&1 || echo %%R>>"%GITIGNORE%"
  )
)

:: 5) 빈 폴더 구조 보존(.gitkeep)
for %%D in (Data Functions Figures Jupyter Notebooks Model Output Python x13as) do (
  if exist "%%D" (
    if not exist "%%D\.gitkeep" echo.> "%%D\.gitkeep"
  )
)

:: 6) 필수 파일 존재 확인
if not exist "colab_setup.py" (
  echo [WARN] colab_setup.py가 없습니다. (루트에 위치 권장)
)

:: 7) 변경 내역 미리보기
echo.
echo [INFO] 변경 파일 목록:
git status --short
echo.
set /p _go="이 변경을 커밋/푸시할까? (Y/N): "
if /i not "%_go%"=="Y" (
  echo 취소함.
  pause
  exit /b 0
)

:: 8) 스테이징/커밋
git add -A
for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set TODAY=%%a-%%b-%%c
set NOW=%time: =0%
set NOW=%NOW::=_%
git commit -m "Auto-commit: %TODAY% %NOW%" 2>nul
if errorlevel 1 (
  echo [INFO] 커밋할 변경 사항 없음.
) else (
  echo [OK] 커밋 완료.
)

:: 9) 푸시
echo [INFO] GitHub로 푸시 중...
git push -u origin %DEFAULT_BRANCH%
if errorlevel 1 (
  echo.
  echo [ERROR] 푸시 실패.
  echo - GitHub 저장소/권한/토큰 확인 필요.
  pause
  exit /b 1
)

echo.
echo =========================================================
echo   완료! GitHub: %REPO_URL%
echo =========================================================
pause