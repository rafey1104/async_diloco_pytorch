bash_script = """#!/bin/bash

# Stop on first error
set -e

echo "🔧 Creating Conda environment..."
conda create -n async-diloco-env python=3.10 -y
echo "✅ Environment created."

echo "🔁 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate async-diloco-env

echo "📦 Installing Python packages..."
pip install torch torchvision matplotlib

echo "🚀 All done. To get started:"
echo "1. Paste the respective files into the folders."
echo "2. Run: conda activate async-diloco-env"
echo "3. Then: python main.py"
"""
