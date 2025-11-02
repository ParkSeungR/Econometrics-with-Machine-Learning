@echo off
echo ====================================
echo  개발환경 설정 (Python 3.10 버전 전용)
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

REM WinPython의 pip를 명시적으로 사용 (3.10.11 버전 고정)
set "PYTHON_BIN=WinPython\python-3.10.11.amd64\Scripts\pip.exe"

if not exist "%PYTHON_BIN%" (
    echo [ERROR] Python 3.10.11 환경을 찾을 수 없습니다.
    echo WinPython을 다운로드하여 정확히 "WinPython\python-3.10.11.amd64" 경로에 위치시키세요.
    pause
    exit
)

"%PYTHON_BIN%" install -r "%REQ_FILE%" --no-warn-script-location

echo.
echo ====================================
echo  모든 라이브러리 설치 완료! (Python 3.10)
echo ====================================
pause