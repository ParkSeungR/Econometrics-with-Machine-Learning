@echo off
echo ====================================
echo  개발환경 설정 (한번만 실행)
echo ====================================
echo.

cd /d "%~dp0"

REM requirements 파일 확인
if exist "requirements.txt" (
    set REQ_FILE=requirements.txt
    echo requirements.txt 파일을 찾았습니다.
) else if exist "requirements" (
    set REQ_FILE=requirements
    echo requirements 파일을 찾았습니다.
) else (
    echo requirements 파일이 없습니다!
    echo 현재 폴더의 파일들:
    dir *.txt
    pause
    exit
)

echo.
echo %REQ_FILE%의 모든 라이브러리를 설치합니다...
echo WinPython 환경을 사용합니다...
echo.

REM WinPython의 pip를 명시적으로 사용
"WinPython\python-3.10.11.amd64\Scripts\pip.exe" install -r "%REQ_FILE%" --no-warn-script-location

echo.
echo ====================================
echo  모든 라이브러리 설치 완료!
echo  이제 USB를 복사해서 배포하세요.
echo ====================================
pause