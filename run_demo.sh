#!/bin/bash

# Run the OpenEvolve with OpenRouter demo
# This script uses the OPENROUTER_API_KEY from environment variables

# Check if OPENROUTER_API_KEY is set
if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "Warning: OPENROUTER_API_KEY environment variable is not set."
  echo "The demo will still run, but you may need to set this variable for API access."
fi

# Run the demo
python simple_demo.py

# Print instructions for the web interface
echo ""
echo "To run the web interface:"
echo "python web_interface.py"