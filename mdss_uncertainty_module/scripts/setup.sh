#!/bin/bash
set -e

echo "🚀 Setting up MDSS Uncertainty Module..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .
pip install -e ".[dev]"

# Create directories
mkdir -p data/{raw,processed,results,plots}
mkdir -p demo/sample_xrays
mkdir -p logs

# Download CheXpert small validation set
echo "Downloading data..."
python scripts/download_data.py

echo "✅ Setup complete!"
echo "Activate environment: source .venv/bin/activate"
echo "Start API: make run-api"
echo "Run overnight batch: make overnight"
