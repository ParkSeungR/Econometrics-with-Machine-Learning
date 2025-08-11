@echo off
setlocal

:: ===== 사용자 설정 =====
set "GITHUB_USER=ParkSeungR"
set "REPO_NAME=Econometrics-with-Machine-Learning"
set "BRANCH=main"
:: =======================

echo.
echo =========================================================
echo   GitHub Push - %REPO_NAME% (%BRANCH%)
echo =========================================================

:: Git 설치 확인
git --version >nul 2>&1 || (
  echo [ERROR] Git 미설치. https://git-scm.com/download/win 설치 후 재시도.
  pause & exit /b 1
)

:: 저장소 초기화/원격 설정
if not exist ".git" git init
set "REPO_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git"
git remote get-url origin >nul 2>&1 || git remote add origin "%REPO_URL%"
git checkout -q -B %BRANCH%

:: (한 번만) 인증 도우미 사용: 첫 푸시 때 아이디/토큰 입력 후 저장됨
git config --global credential.helper manager

:: 원격 최신 반영
echo [STEP] pull --rebase
git fetch origin
git pull --rebase --autostash origin %BRANCH% || (
  echo [ERROR] pull 중 충돌. 파일 수정→ git add → git rebase --continue 후 다시 실행.
  pause & exit /b 1
)

:: 변경 사항 스테이징
git add -A

:: 변경이 없으면 커밋 스킵
git diff --cached --quiet
if %errorlevel%==0 (
  echo [INFO] 커밋할 변경 없음.
) else (
  git commit -m "Auto-commit"
  if errorlevel 1 (
    echo [ERROR] 커밋 실패.
    pause & exit /b 1
  )
)

:: 푸시
echo [STEP] push
git rev-parse --abbrev-ref --symbolic-full-name @{u} >nul 2>&1
if errorlevel 1 (
  git push -u origin %BRANCH% || ( echo [ERROR] 푸시 실패. 권한/토큰 확인. & pause & exit /b 1 )
) else (
  git push || ( echo [ERROR] 푸시 실패. 권한/토큰 확인. & pause & exit /b 1 )
)

echo [OK] 동기화 완료 → %REPO_URL%
pause