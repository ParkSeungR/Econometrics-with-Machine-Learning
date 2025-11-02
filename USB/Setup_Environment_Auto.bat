@echo off
echo ====================================
echo  개발환경 설정 (최신 버전 자동 탐색)
echo ====================================
echo.

cd /d "%~dp0"

REM requirements 파일 확인 (생략)
if exist "requirements.txt" (
    set REQ_FILE=requirements.txt
    echo requirements.txt 파일을 찾았습니다.
) else if exist "requirements" (
    set REQ_FILE=requirements
    echo requirements 파일을 찾았습니다.
) else (
    echo [ERROR] requirements.txt 파일이 없습니다.
    pause
    exit
)

echo.
echo %REQ_FILE%의 모든 라이브러리를 설치합니다...
echo WinPython 환경을 사용합니다...
echo.

REM WinPython 폴더(WinPython\python-*.amd64) 경로 자동 탐색
set "PYTHON_PATH="
for /d %%d in ("WinPython\python-*") do (
    if not defined PYTHON_PATH set "PYTHON_PATH=%%d"
)

if not defined PYTHON_PATH (
    echo [ERROR] WinPython 폴더(WinPython\python-*.amd64)를 찾을 수 없습니다.
    echo WinPython을 다운로드하여 폴더 안에 위치시켰는지 확인하세요.
    pause
    exit
)

REM WinPython의 pip를 명시적으로 사용
"%PYTHON_PATH%\Scripts\pip.exe" install -r "%REQ_FILE%" --no-warn-script-location

echo.
echo ====================================
echo  모든 라이브러리 설치 완료!
echo  (버전 충돌시 3.10 전용 파일 사용 권장)
echo ====================================
pause