bash <<'SETUP' || printf '\nSetup failed; check the log shown above.\n'
set -Eeuo pipefail

LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/standalone-pipeline-setup.XXXXXX.log")"
exec > >(tee "$LOG_FILE") 2>&1

finish() {
    local status=$?
    trap - EXIT

    printf '\nSetup exit code: %s\nLog: %s\n' "$status" "$LOG_FILE"
    if [[ "$status" -eq 0 ]]; then
        printf 'Setup completed. Activate with: conda activate standalone-pipeline\n'
    fi

    read -r -p "Press Enter to return to your terminal..." </dev/tty || true
    exit "$status"
}

trap finish EXIT
trap 'printf "\nERROR at line %s (exit code %s).\n" "$LINENO" "$?" >&2' ERR

# Locate the latest project.
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
if [[ ! -f "$PROJECT_DIR/requirements-expressive.txt" ]]; then
    read -r -p "Full Linux path to standalone_pipeline: " PROJECT_DIR </dev/tty
fi
cd "$PROJECT_DIR"

for file in requirements.txt requirements-tts.txt \
            requirements-expressive.txt requirements-dev.txt; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing project file: %s/%s\n' "$PWD" "$file" >&2
        exit 1
    fi
done

command -v conda >/dev/null
command -v apt-get >/dev/null
nvidia-smi

# Initialize Conda; disable nounset while running activation scripts.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

# Remove the old environment if it exists.
if conda env list | awk '
    $1 == "standalone-pipeline" { found = 1 }
    END { exit !found }
'; then
    conda env remove --name standalone-pipeline --yes
fi

# Create the environment with CUDA development tools and GCC/G++ 12.
conda create --name standalone-pipeline --yes \
    --override-channels \
    -c nvidia/label/cuda-12.1.1 \
    -c conda-forge \
    python=3.10 pip "cuda-toolkit=12.1" "gxx_linux-64=12"

conda activate standalone-pipeline
set -u

# System dependencies.
sudo apt-get update
sudo apt-get install -y \
    git ffmpeg espeak-ng sox libsox-dev libsndfile1 \
    build-essential cmake pkg-config

export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDAHOSTCXX="${CXX:?Conda C++ compiler was not activated}"

# Preserve compatibility with the pinned Whisper build.
python -m pip install --upgrade pip wheel "setuptools<81"

# CUDA-enabled PyTorch.
python -m pip install \
    torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Build the LLM runtime with CUDA support.
CMAKE_ARGS="-DGGML_CUDA=ON" \
CMAKE_BUILD_PARALLEL_LEVEL=2 \
python -m pip install \
    --no-cache-dir \
    --no-binary=llama-cpp-python \
    -r requirements.txt

python -m pip install --no-build-isolation openai-whisper==20231117

# Install both TTS backends and development requirements.
python -m pip install \
    -r requirements-expressive.txt \
    -r requirements-dev.txt

python -m spacy download en_core_web_sm
python -m pip check

# Provision prototype models and verify the completed setup.
python download_models.py --phase prototype --tts kokoro
python run.py prepare-reference
python check_environment.py --config config.local.json
SETUP
