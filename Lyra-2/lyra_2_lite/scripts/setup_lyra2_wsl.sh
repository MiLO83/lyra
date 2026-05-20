#!/usr/bin/env bash
# Lyra 2 full setup inside WSL2 Ubuntu 24.04.
# Adapted from Lyra-2/INSTALL.md for WSL.
# Idempotent: each step checks if already done.

set -eo pipefail   # NOT -u: conda's activate/deactivate scripts reference unset vars
exec > >(tee -a /home/milo/lyra2_setup.log) 2>&1
echo "=== Lyra 2 setup starting $(date) ==="
# Pre-initialise conda's expected backup vars so its deactivate scripts don't trip set -e
export CONDA_BACKUP_CXX=""
export CONDA_BACKUP_CC=""

cd /home/milo

# --- 1. Miniconda ---
if [ ! -d /home/milo/miniconda3 ]; then
    echo "[$(date +%H:%M:%S)] Installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /home/milo/miniconda3
    rm miniconda.sh
fi
export PATH=/home/milo/miniconda3/bin:$PATH
source /home/milo/miniconda3/etc/profile.d/conda.sh
echo "[$(date +%H:%M:%S)] conda: $(conda --version)"

# Conda 26.x requires explicit acceptance of Anaconda channel ToS.
# We accept once and configure conda-forge as the priority channel.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
conda config --add channels conda-forge
conda config --set channel_priority strict

# --- 2. lyra2 env ---
if ! conda env list | grep -q '^lyra2 '; then
    echo "[$(date +%H:%M:%S)] Creating lyra2 env..."
    conda create -n lyra2 python=3.10 pip cmake ninja libgl ffmpeg packaging -c conda-forge --override-channels -y
fi
conda activate lyra2
echo "[$(date +%H:%M:%S)] python: $(python --version)"

# --- 3. Build toolchain ---
if ! conda list -n lyra2 2>/dev/null | grep -q '^gcc '; then
    echo "[$(date +%H:%M:%S)] Installing gcc/gxx/eigen/zlib..."
    CONDA_BACKUP_CXX="" conda install -n lyra2 gcc=13.3.0 gxx=13.3.0 eigen zlib -c conda-forge -y
fi

# --- 4. CUDA 12.8 toolkit ---
if [ ! -f "$CONDA_PREFIX/bin/nvcc" ]; then
    echo "[$(date +%H:%M:%S)] Installing CUDA 12.8 toolkit (large, ~5 min)..."
    conda install -n lyra2 cuda -c nvidia/label/cuda-12.8.0 -y
fi
export CUDA_HOME=$CONDA_PREFIX
echo "[$(date +%H:%M:%S)] nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1)"

# --- 5. PyTorch 2.7.1 + cu128 ---
if ! python -c 'import torch; assert torch.__version__.startswith("2.7")' 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] Installing PyTorch 2.7.1 cu128..."
    pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128
fi
python -c 'import torch; print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())'

# --- 6. Env vars for builds ---
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# --- 7. requirements.txt (no-deps to avoid version drift) ---
LYRA2_DIR=/mnt/c/Users/rxcam/Documents/lyra_pr_branch/Lyra-2
echo "[$(date +%H:%M:%S)] Installing requirements.txt..."
pip install --no-deps -r "$LYRA2_DIR/requirements.txt"

# --- 8. MoGe ---
if ! python -c 'import moge.model.v1' 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] Installing MoGe from github..."
    pip install 'git+https://github.com/microsoft/MoGe.git'
fi

# --- 9. transformer_engine (SLOW: ~20-30 min) ---
if ! python -c 'import transformer_engine.pytorch' 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] Building transformer_engine (slow, ~20-30 min)..."
    pip install --no-build-isolation 'transformer_engine[pytorch]'
fi
# Symlink cuda_runtime as cudart for transformer_engine compatibility
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"

# --- 10. flash-attn 2.6.3 (SLOW: ~30-45 min) ---
if ! python -c 'import flash_attn' 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] Building flash-attn 2.6.3 (slow, ~30-45 min)..."
    MAX_JOBS=16 pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3
fi

# --- 11. vipe CUDA extension ---
if ! python -c 'import vipe_ext' 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] Building vipe extension..."
    USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e "$LYRA2_DIR/lyra_2/_src/inference/vipe"
fi

# --- 12. depth_anything_3 extension ---
if ! python -c 'import depth_anything_3.api' 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] Building depth_anything_3 (with gs)..."
    pip install --no-build-isolation -e "$LYRA2_DIR/lyra_2/_src/inference/depth_anything_3[gs]"
fi

# --- 13. bitsandbytes (for our int8 path) ---
if ! python -c 'import bitsandbytes' 2>/dev/null; then
    pip install bitsandbytes
fi

# --- 14. Verify everything ---
echo "[$(date +%H:%M:%S)] Verifying all imports..."
python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1, bitsandbytes
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('flash_attn:', flash_attn.__version__)
print('transformer_engine: OK')
print('vipe_ext: OK')
print('depth_anything_3: OK')
print('moge: OK')
print('bitsandbytes:', bitsandbytes.__version__)
print('=== ALL IMPORTS OK ===')
"

echo "[$(date +%H:%M:%S)] === Lyra 2 setup COMPLETE ==="
echo "Next: cd $LYRA2_DIR && PYTHONPATH=. python -m lyra_2._src.inference.lyra2_custom_traj_inference --low-vram int8 ..."
