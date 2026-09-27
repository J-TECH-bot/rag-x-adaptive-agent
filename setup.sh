#!/usr/bin/env bash

# RAG-X Adaptive Agent
# One-command environment setup
#
# Usage:
#   source setup.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "RAG-X Adaptive Agent - Environment Setup"
echo "=============================================="

# ------------------------------------------------
# 1. Find Python 3.13
# ------------------------------------------------

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD="$PYTHON_BIN"
elif command -v python3.13 >/dev/null 2>&1; then
    PYTHON_CMD="python3.13"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo "ERROR: Python 3 was not found."
    echo "Install Python 3.13 and run this script again."
    return 1 2>/dev/null || exit 1
fi

PYTHON_VERSION="$("$PYTHON_CMD" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"

echo "Detected Python: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != 3.13.* ]]; then
    echo ""
    echo "ERROR: RAG-X requires Python 3.13.x"
    echo "Detected: Python $PYTHON_VERSION"
    echo ""
    echo "If Python 3.13 is installed, run:"
    echo "PYTHON_BIN=python3.13 source setup.sh"
    return 1 2>/dev/null || exit 1
fi

echo "Using Python: $PYTHON_CMD"

# ------------------------------------------------
# 2. Check existing virtual environment
# ------------------------------------------------

if [[ -d ".venv" ]]; then

    VENV_VERSION="$(
        .venv/bin/python -c \
        'import sys; print(".".join(map(str, sys.version_info[:3])))' \
        2>/dev/null || true
    )"

    if [[ "$VENV_VERSION" != 3.13.* ]]; then
        echo ""
        echo "Existing .venv uses Python $VENV_VERSION"
        echo "Recreating .venv with Python 3.13..."
        rm -rf .venv
    else
        echo "Existing Python 3.13 virtual environment found."
    fi
fi

# ------------------------------------------------
# 3. Create virtual environment
# ------------------------------------------------

if [[ ! -d ".venv" ]]; then
    echo ""
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv .venv
fi

# ------------------------------------------------
# 4. Activate virtual environment
# ------------------------------------------------

source .venv/bin/activate

echo ""
echo "Virtual environment activated:"
echo "$VIRTUAL_ENV"

# ------------------------------------------------
# 5. Upgrade packaging tools
# ------------------------------------------------

echo ""
echo "Upgrading pip, setuptools and wheel..."

python -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------
# 6. Install project dependencies
# ------------------------------------------------

echo ""
echo "Installing RAG-X dependencies..."
echo "This may take some time because PyTorch and"
echo "Sentence Transformers are large packages."
echo ""

python -m pip install -r requirements.txt

# ------------------------------------------------
# 7. Validate imports
# ------------------------------------------------

echo ""
echo "=============================================="
echo "Validating installed packages"
echo "=============================================="

python - <<'PY'
import sys

import torch
import transformers
import sentence_transformers
import faiss
import pypdf
import rank_bm25
import numpy
import pandas
import sklearn

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Sentence Transformers:", sentence_transformers.__version__)
print("FAISS:", faiss.__version__)
print("pypdf:", pypdf.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("rank-bm25: import OK")
PY

# ------------------------------------------------
# 8. Check dependency consistency
# ------------------------------------------------

echo ""
echo "Running pip check..."

python -m pip check

# ------------------------------------------------
# 9. Finish
# ------------------------------------------------

echo ""
echo "=============================================="
echo "RAG-X environment is ready."
echo "=============================================="
echo ""
echo "Virtual environment:"
echo "$VIRTUAL_ENV"
echo ""
echo "Run project scripts with:"
echo ""
echo "  PYTHONPATH=src python <script>"
echo ""
