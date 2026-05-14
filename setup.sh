#!/bin/bash
# Setup script for Compute Engine Ubuntu 22.04 (Jammy)
# Run: bash setup.sh

echo "=== Setup TFX Pipeline Environment ==="

# Buat conda environment
conda create -n tfx-submission python==3.9.15 -y
conda activate tfx-submission

# Install dependencies
pip install jupyter scikit-learn tensorflow tfx==1.11.0 flask joblib keras-tuner

echo ""
echo "=== Setup selesai! ==="
echo ""
echo "To run the notebook:"
echo "  conda activate tfx-submission"
echo "  cd /path/to/submission"
echo "  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser"
