@echo off 
cd /d "%~dp0" 
call envs\ml_econ_env\Scripts\activate.bat 
python -m jupyter notebook 
