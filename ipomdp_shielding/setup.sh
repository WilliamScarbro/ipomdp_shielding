#!/bin/bash
# Setup script for ipomdp_shielding project

set -e  # Exit on error

echo "Setting up ipomdp_shielding virtual environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment in ./venv..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate when done, run:"
echo "  deactivate"
echo ""
