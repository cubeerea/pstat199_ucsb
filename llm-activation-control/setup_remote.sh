#!/usr/bin/env bash
# One-shot setup script for the llm-activation-control jailbreaking experiment.
# Run this on your Linux+CUDA remote machine from inside this directory:
#
#   cd llm-activation-control/
#   bash setup_remote.sh [VLLM_FORK_URL]
#
# The optional argument is the GitHub URL of the authors' custom vLLM fork.
# If omitted you'll be prompted.  The fork must expose:
#   vllm.control_vectors.request.ControlVectorRequest
#
# Prerequisites:
#   - conda (miniconda/mambaforge) — will suggest install command if missing
#   - nvidia-smi accessible, CUDA ≥ 12.1
#   - HF token with Gemma-2 + Llama-3 access already logged in, OR
#     HUGGING_FACE_HUB_TOKEN env var set before calling this script

set -euo pipefail

VLLM_FORK_URL="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAC_DIR="$REPO_ROOT/llm-activation-control"

# ── helpers ──────────────────────────────────────────────────────────────────

info()  { echo "[setup] $*"; }
warn()  { echo "[setup][WARN] $*" >&2; }
die()   { echo "[setup][ERROR] $*" >&2; exit 1; }

check_cuda() {
    if ! command -v nvidia-smi &>/dev/null; then
        die "nvidia-smi not found. This experiment requires an NVIDIA GPU with CUDA ≥ 12.1."
    fi
    local cuda_ver
    cuda_ver=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
    info "Detected CUDA $cuda_ver"
    local major=${cuda_ver%%.*}
    if [[ "$major" -lt 12 ]]; then
        warn "CUDA $cuda_ver detected but vLLM 0.8.x requires ≥ 12.1. Build from source or upgrade driver."
    fi
}

ensure_conda() {
    if ! command -v conda &>/dev/null; then
        die "conda not found. Install Miniconda first:
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p \$HOME/miniconda3
  source \$HOME/miniconda3/etc/profile.d/conda.sh
  conda init bash && exec \$SHELL
Then re-run this script."
    fi
    # Make `conda activate` work inside bash scripts
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    info "conda $(conda --version)"
}

# ── main ─────────────────────────────────────────────────────────────────────

info "=== PID Steering / LLM Jailbreak Setup ==="
check_cuda
ensure_conda

# ── env 1: angular_steering ───────────────────────────────────────────────────

if conda env list | grep -q "^angular_steering "; then
    info "Conda env 'angular_steering' already exists — skipping creation."
else
    info "Creating conda env 'angular_steering' (Python 3.10)..."
    conda create -n angular_steering python=3.10 -y
fi

conda activate angular_steering

info "Installing pip requirements..."
pip install -r "$LAC_DIR/requirements.txt"

info "Installing local llm_activation_control package..."
pip install -e "$LAC_DIR"

# ── vLLM fork ─────────────────────────────────────────────────────────────────

VLLM_DIR="$REPO_ROOT/vllm"

if python -c "from vllm.control_vectors.request import ControlVectorRequest" 2>/dev/null; then
    info "Custom vLLM fork already installed — skipping."
else
    if [[ -z "$VLLM_FORK_URL" ]]; then
        echo ""
        echo "  The experiment requires the authors' custom vLLM fork (with control_vectors support)."
        echo "  Its URL is not documented in the repo. Try:"
        echo "    https://github.com/dungnvus   (or dungnvnus)  — look for a 'vllm' repo"
        echo "    https://github.com/HieuMVu/AngularSteering    — upstream; may link the fork"
        echo "    OpenReview page for 'Activation Steering with a Feedback Controller'"
        echo "    Contact the authors (Dung V. Nguyen)"
        echo ""
        read -rp "  vLLM fork URL (or press Enter to skip and install upstream vLLM instead): " VLLM_FORK_URL
    fi

    if [[ -n "$VLLM_FORK_URL" ]]; then
        if [[ -d "$VLLM_DIR/.git" ]]; then
            info "vllm/ directory already cloned — skipping git clone."
        else
            info "Cloning vLLM fork into $VLLM_DIR ..."
            git clone "$VLLM_FORK_URL" "$VLLM_DIR"
        fi
        info "Installing vLLM fork (editable, precompiled wheels)..."
        VLLM_USE_PRECOMPILED=1 pip install --editable "$VLLM_DIR"
        python -c "from vllm.control_vectors.request import ControlVectorRequest; print('[setup] vLLM fork OK')"
    else
        warn "Skipping custom vLLM fork. Installing upstream vLLM==0.8.5.post1 as a placeholder."
        warn "generate_responses.py and endpoint.py will fail until the fork is installed."
        pip install "vllm==0.8.5.post1"
    fi
fi

conda deactivate

# ── env 2: lm_eval (TinyBenchmarks) ──────────────────────────────────────────

if conda env list | grep -q "^lm_eval "; then
    info "Conda env 'lm_eval' already exists — skipping creation."
else
    info "Creating conda env 'lm_eval' (Python 3.10)..."
    conda create -n lm_eval python=3.10 -y
fi

conda activate lm_eval

LM_EVAL_DIR="$REPO_ROOT/lm-evaluation-harness"
if [[ -d "$LM_EVAL_DIR" ]]; then
    info "lm-evaluation-harness already cloned."
else
    info "Cloning EleutherAI/lm-evaluation-harness..."
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness "$LM_EVAL_DIR"
fi
pip install -e "$LM_EVAL_DIR"

conda deactivate

# ── HF auth check ─────────────────────────────────────────────────────────────

conda activate angular_steering

if python -c "import huggingface_hub; huggingface_hub.whoami()" 2>/dev/null; then
    info "HF credentials found."
else
    warn "Not logged in to Hugging Face. Run: huggingface-cli login"
    warn "(Or set HUGGING_FACE_HUB_TOKEN before this script.)"
fi

# ── smoke-test import ─────────────────────────────────────────────────────────

python - <<'PY'
import sys
fails = []

try:
    import torch
    print(f"  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    fails.append(f"torch: {e}")

try:
    import transformers
    print(f"  transformers {transformers.__version__}")
except Exception as e:
    fails.append(f"transformers: {e}")

try:
    import datasets
    print(f"  datasets {datasets.__version__}")
except Exception as e:
    fails.append(f"datasets: {e}")

try:
    from vllm.control_vectors.request import ControlVectorRequest
    print("  vLLM fork (control_vectors) OK")
except Exception as e:
    fails.append(f"vllm fork: {e}")

if fails:
    print("\n[setup][WARN] Some imports failed:", file=sys.stderr)
    for f in fails:
        print(f"  - {f}", file=sys.stderr)
else:
    print("[setup] All core imports OK")
PY

conda deactivate

# ── summary ──────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════"
echo " Setup complete. Next steps:"
echo ""
echo " 1. Run angular_steering.ipynb (conda activate angular_steering) to"
echo "    generate output/ (steering directions per model/method)."
echo "    Set METHOD_PREFIX = \"PID_\" and start with Qwen/Qwen2.5-3B-Instruct."
echo ""
echo " 2. Edit generate_responses.py:"
echo "      model_ids = [\"Qwen/Qwen2.5-3B-Instruct\"]"
echo "      METHOD_PREFIX = \"PID_\""
echo "    Then: python generate_responses.py"
echo ""
echo " 3. Run eval (substring matching, no server needed):"
echo "      python evaluate_jailbreak.py   # set methods=[\"substring_matching\"]"
echo ""
echo " 4. For LlamaGuard 3 eval:"
echo "      GPU=0 bash eval.sh &"
echo "      python evaluate_jailbreak.py   # set methods=[\"LlamaGuard 3\"]"
echo ""
echo " 5. For TinyBenchmarks:"
echo "      python endpoint.py Qwen/Qwen2.5-3B-Instruct  # in angular_steering env"
echo "      conda activate lm_eval && GPU=0 bash eval_tinybench.sh"
echo "══════════════════════════════════════════════════════════"
