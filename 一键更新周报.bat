@echo off
echo ========================================================
echo        Starting Market Monitor Report
echo ========================================================
echo.

:: 1. Set your specific Python path
set PYTHON_PATH=C:\ProgramData\anaconda3\python.exe

:: 2. Verify Python exists
if not exist "%PYTHON_PATH%" (
    echo [ERROR] Python executable not found at:
    echo %PYTHON_PATH%
    echo.
    echo Please verify your Anaconda installation path.
    pause
    exit
)

:: 3. Navigate to the Code directory
cd /d "%~dp02_Code"

:: 4. Run the Python script
echo [INFO] Processing data with Python...
"%PYTHON_PATH%" build_report.py

:: 5. Check result and open report
if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Report generated successfully!
    echo Opening report in your default browser...
    start "" "%~dp0Report_Output\index.html"
) else (
    echo.
    echo [FAILURE] The script encountered an error.
    echo Please check the error messages above.
)

pause