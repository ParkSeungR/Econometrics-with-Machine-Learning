@echo off
setlocal

:: ===== 사용자 설정 =====
set "GITHUB_USER=ParkSeungR"
set "REPO_NAME=Econometrics-with-Machine-Learning"
set "BRANCH=main"
set "STRICT_SYNC=1"   :: 1=원격을 로컬과 완전히 동일하게(로컬에 없는 파일은 원격에서 삭제)
:: =======================

echo.
echo =========================================================
echo   GitHub Push - %REPO_NAME% (%BRANCH%)
echo =========================================================

:: Git 확인
git --version >nul 2>&1 || ( echo [ERROR] Git 미설치. https://git-scm.com/download/win ; pause & exit /b 1 )

:: 저장소 초기화/원격
if not exist ".git" git init
set "REPO_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git"
git remote get-url origin >nul 2>&1 || git remote add origin "%REPO_URL%"
git config --global credential.helper manager
git checkout -q -B %BRANCH%

:: 원격 최신 반영
echo [STEP] pull --rebase
git fetch origin
git pull --rebase --autostash origin %BRANCH% || (
  echo [ERROR] pull 충돌. 파일 정리 후 git add → git rebase --continue ; pause & exit /b 1
)

:: === STRICT 모드: 인덱스 초기화 후 현재 로컬 스냅샷만 재등록 ===
if "%STRICT_SYNC%"=="1" (
  echo [STEP] strict sync: repo index reset to local snapshot
  git rm -r --cached . >nul 2>&1
)

:: 변경 스테이징(삭제 포함)
git add -A

:: 커밋(변경 없으면 스킵)
git diff --cached --quiet
if %errorlevel%==0 (
  echo [INFO] 커밋할 변경 없음.
) else (
  for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set TODAY=%%a-%%b-%%c
  set NOW=%time: =0%
  set NOW=%NOW::=_%
  git commit -m "Auto-commit (strict sync): %TODAY% %NOW%" || ( echo [ERROR] 커밋 실패 ; pause & exit /b 1 )
)

:: 푸시
echo [STEP] push
git rev-parse --abbrev-ref --symbolic-full-name @{u} >nul 2>&1
if errorlevel 1 ( git push -u origin %BRANCH% ) else ( git push ) || ( echo [ERROR] 푸시 실패. 권한/토큰 확인 ; pause & exit /b 1 )

echo [OK] 동기화 완료 → %REPO_URL%
pause