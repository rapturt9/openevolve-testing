#!/bin/bash

# Setup script for OpenRouter-OpenEvolve
# This script installs OpenEvolve and all dependencies

set -e  # Exit on error

echo "Setting up OpenRouter-OpenEvolve..."

# Check if OpenEvolve is already installed
if [ -d "openevolve" ]; then
  echo "OpenEvolve directory already exists. Updating..."
  cd openevolve
  git pull
  pip install -e .
  cd ..
else
  echo "Cloning OpenEvolve repository..."
  git clone https://github.com/codelion/openevolve.git
  cd openevolve
  pip install -e .
  cd ..
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "WARNING: OPENROUTER_API_KEY environment variable is not set."
  echo "You will need to set this variable to run the examples:"
  echo "export OPENROUTER_API_KEY=your_openrouter_api_key"
else
  echo "OpenRouter API key found in environment variables."
fi

# Create output directory
mkdir -p output

echo "Setup complete! You can now run the examples:"
echo "  - Run the demo: ./run_demo.sh"
echo "  - Run an experiment: python run_experiment.py --iterations 50"
echo "  - Start the web interface: python web_interface.py"