#!/bin/bash
# Setup script for Ubuntu 22.04 (Jammy)
# Run: bash setup.sh

echo "=== Setup TFX Pipeline Environment ==="

# Create conda environment
conda create -n adult-income-tfx python==3.9.15 -y
conda activate adult-income-tfx

# Install dependencies
pip install jupyter scikit-learn tensorflow tfx==1.11.0 flask joblib keras-tuner

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the notebook:"
echo "  conda activate adult-income-tfx"
echo "  cd /path/to/project"
echo "  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser"
