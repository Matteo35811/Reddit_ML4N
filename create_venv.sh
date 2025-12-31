#!/usr/bin/env bash

set -e  # stop on errors

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Install it before proceeding."
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate venv
echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install requirements if needed
if [ -f "$REQUIREMENTS" ]; then
    echo "Checking installed packages..."
    installed=$(pip list --format=freeze | wc -l)

    if [ "$installed" -le 1 ]; then
        echo "Installing requirements..."
        pip install --upgrade pip
        pip install -r "$REQUIREMENTS"
    else
        echo "Requirements already installed."
    fi
else
    echo "requirements.txt not found. Skipping installation."
fi

echo "Virtual environment is ready."
python --version
