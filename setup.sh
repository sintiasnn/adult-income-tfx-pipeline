#!/bin/bash
# Setup script for Ubuntu 22.04 (Jammy)
# Run: bash setup.sh

set -e

echo "=== Setup TFX Pipeline Environment ==="

# Install Miniconda if not available
if ! command -v conda &> /dev/null; then
    echo "Miniconda not found. Downloading and installing..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.7.1-0-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda
    source $HOME/miniconda/bin/activate
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
fi

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
