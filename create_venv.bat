@echo off
set VENV_DIR=.venv
set REQUIREMENTS=requirements.txt

echo Checking Python installation...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found in PATH.
    exit /b 1
)

REM Create venv if missing
if not exist %VENV_DIR% (
    echo Creating virtual environment in %VENV_DIR% ...
    python -m venv %VENV_DIR%
) else (
    echo Virtual environment already exists.
)

REM Activate
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

REM Install requirements if needed
if exist %REQUIREMENTS% (
    echo Checking installed packages...
    pip list >nul 2>&1

    if %errorlevel% neq 0 (
        echo Installing requirements...
        pip install --upgrade pip
        pip install -r %REQUIREMENTS%
    ) else (
        echo Requirements already installed.
    )
) else (
    echo requirements.txt not found.
)

echo Virtual environment ready.
python --version
