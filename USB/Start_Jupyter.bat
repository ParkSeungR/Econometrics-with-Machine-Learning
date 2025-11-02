@echo off
echo ====================================
echo  계량경제학 ^& 머신러닝 분석환경
echo ====================================
echo.

cd /d "%~dp0"

echo Jupyter Lab을 시작합니다...
echo.

REM 직접 명령행 옵션으로 토큰/패스워드 비활성화
start /B "Jupyter Lab" "%~dp0WinPython\python-3.10.11.amd64\python.exe" -m jupyterlab --port=8888 --ServerApp.token="" --ServerApp.password="" --ServerApp.allow_origin="*" --ServerApp.disable_check_xsrf=True --no-browser

REM 서버 시작 대기
echo 서버 시작 중... 잠시만 기다려주세요.
timeout /t 5 >nul

REM 브라우저 자동 실행
echo 브라우저를 엽니다...
start http://localhost:8888

echo.
echo ====================================
echo http://localhost:8888 에서 즉시 사용 가능
echo 종료하려면 이 창을 닫으세요
echo ====================================

pause