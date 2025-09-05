set "CURRENT_DIR=%~dp0"
set PYTHONPATH=%CURRENT_DIR%;%PYTHONPATH%
start /b python.exe "%CURRENT_DIR%\run_DeePFAS.py"
