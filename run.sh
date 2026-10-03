#!/usr/bin/env bash
set -e -o pipefail

# CHANGE THIS to your Linux project path.
cd "/absolute/path/to/standalone_pipeline"
test -f requirements.txt
test -f requirements-expressive.txt
test -f requirements-dev.txt

# Check the NVIDIA driver before removing the environment.
nvidia-smi >/dev/null

# Initialize Conda and leave the environment being removed.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

if conda env list | awk '
    $1 == "standalone-pipeline" { found = 1 }
    END { exit !found }
'; then
    conda env remove --name standalone-pipeline --yes
fi

# Python, CUDA development toolkit, and a compatible C++ compiler.
conda create --name standalone-pipeline --yes \
    --override-channels \
    -c nvidia/label/cuda-12.1.1 \
    -c conda-forge \
    python=3.10 pip "cuda-toolkit=12.1" "gxx_linux-64=12"

conda activate standalone-pipeline

sudo apt-get update
sudo apt-get install -y \
    git ffmpeg espeak-ng sox libsox-dev libsndfile1 \
    build-essential cmake pkg-config

export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDAHOSTCXX="${CXX:?Conda C++ compiler was not activated}"

# The pinned Whisper release still needs pkg_resources.
python -m pip install --upgrade pip wheel "setuptools<81"

# Install GPU-enabled PyTorch first.
python -m pip install \
    torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Compile llama-cpp-python with CUDA support.
CMAKE_ARGS="-DGGML_CUDA=ON" \
CMAKE_BUILD_PARALLEL_LEVEL=2 \
python -m pip install \
    --no-cache-dir \
    --no-binary=llama-cpp-python \
    -r requirements.txt

# Install Whisper without its incompatible isolated build environment.
python -m pip install --no-build-isolation openai-whisper==20231117

# All remaining runtime and development requirements.
python -m pip install \
    -r requirements-expressive.txt \
    -r requirements-dev.txt

python -m spacy download en_core_web_sm
python -m pip check

# Download/reuse prototype weights, prepare the voice, and verify setup.
python download_models.py --phase prototype --tts kokoro
python run.py prepare-reference
python check_environment.py --config config.local.json
