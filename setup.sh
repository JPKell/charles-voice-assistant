#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "== Local Voice Chat — Kokoro / Python 3.13 / spaCy fix =="
echo

# ------------------------------------------------------------
# 1. Ubuntu dependencies
# ------------------------------------------------------------
echo "Installing/checking Ubuntu dependencies..."
sudo apt-get update
sudo apt-get install -y \
    curl \
    git \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
    alsa-utils \
    espeak-ng

# ------------------------------------------------------------
# 2. Python 3.13
# ------------------------------------------------------------
echo
echo "Checking Python 3.13..."

if ! command -v python3.13 >/dev/null 2>&1; then
    echo "python3.13 was not found. Attempting to install it..."
    sudo apt-get install -y python3.13 python3.13-venv
fi

if ! command -v python3.13 >/dev/null 2>&1; then
    echo
    echo "ERROR: Python 3.13 is unavailable."
    exit 1
fi

if ! python3.13 -m venv --help >/dev/null 2>&1; then
    sudo apt-get install -y python3.13-venv
fi

echo "Found: $(python3.13 --version)"

# ------------------------------------------------------------
# 3. Rebuild project environment from scratch
# ------------------------------------------------------------
echo
echo "Recreating .venv with Python 3.13..."
rm -rf .venv
python3.13 -m venv .venv

echo "Virtual environment:"
.venv/bin/python --version

echo
echo "Updating pip/build tools..."
.venv/bin/python -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# 4. Common voice-assistant dependencies
# ------------------------------------------------------------
echo
echo "Installing common voice-assistant dependencies..."
.venv/bin/python -m pip install -r requirements.txt

# CTranslate2 currently requires CUDA 12 cuBLAS and cuDNN 9, while PyTorch may
# use a newer CUDA runtime. Keep the Whisper libraries isolated so both stacks
# can coexist; voice_chat/stt.py preloads only the required shared libraries.
echo
echo "Installing isolated CUDA 12 runtime for Faster-Whisper..."
.venv/bin/python -m pip install --target .venv/cuda12 \
    nvidia-cublas-cu12 \
    "nvidia-cudnn-cu12==9.*"

# ------------------------------------------------------------
# 5. Pin a known Python-3.13 spaCy wheel.
#
# Misaki's broad English extra currently lets pip consider a spaCy
# 4.0 dev source release in some resolver combinations. That source
# path pulls old thinc/blis build requirements and fails on Python 3.13.
#
# spaCy 3.8.15 publishes a CPython 3.13 Linux wheel, so force that wheel.
# ------------------------------------------------------------
echo
echo "Installing spaCy 3.8.15 binary wheel..."
.venv/bin/python -m pip install \
    --only-binary=:all: \
    "spacy==3.8.15"

# ------------------------------------------------------------
# 6. Install only the English dependencies Kokoro actually needs
# for KPipeline(..., trf=False), avoiding spacy-curated-transformers.
# ------------------------------------------------------------
echo
echo "Installing Kokoro/Misaki English runtime dependencies..."
.venv/bin/python -m pip install --upgrade \
    addict \
    regex \
    num2words \
    phonemizer-fork \
    espeakng-loader \
    torch \
    transformers \
    huggingface_hub \
    loguru

# ------------------------------------------------------------
# 7. Install current Misaki and Kokoro source WITHOUT dependency
# resolution. We installed the runtime dependencies explicitly above.
# ------------------------------------------------------------
echo
echo "Installing current Misaki source..."
.venv/bin/python -m pip install --no-deps --upgrade \
    "misaki @ git+https://github.com/hexgrad/misaki.git"

echo
echo "Installing current Kokoro source..."
.venv/bin/python -m pip install --no-deps --upgrade \
    "kokoro @ git+https://github.com/hexgrad/kokoro.git"

# ------------------------------------------------------------
# 8. Pre-install the English spaCy model used by Misaki when trf=False
# ------------------------------------------------------------
echo
echo "Installing spaCy English model used by Misaki..."
.venv/bin/python -m spacy download en_core_web_sm

# ------------------------------------------------------------
# 9. Verify imports and English G2P initialization
# ------------------------------------------------------------
echo
echo "Verifying Python dependencies..."
.venv/bin/python - <<'PY'
import sys

modules = [
    ("faster_whisper", "Faster-Whisper"),
    ("spacy", "spaCy"),
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("misaki", "Misaki"),
    ("kokoro", "Kokoro"),
    ("numpy", "NumPy"),
    ("requests", "Requests"),
    ("sounddevice", "sounddevice"),
    ("soundfile", "soundfile"),
]

failed = []
for module, label in modules:
    try:
        __import__(module)
        print(f"  OK   {label}")
    except Exception as exc:
        failed.append((label, exc))
        print(f"  FAIL {label}: {exc}")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    print("  OK   en_core_web_sm")
except Exception as exc:
    failed.append(("en_core_web_sm", exc))
    print(f"  FAIL en_core_web_sm: {exc}")

if failed:
    print("\nERROR: dependency verification failed.")
    for label, exc in failed:
        print(f"  {label}: {exc}")
    sys.exit(1)

print("\nAll dependency checks passed.")
PY

# ------------------------------------------------------------
# 10. Ollama
# ------------------------------------------------------------
echo
echo "Checking Ollama..."

if command -v ollama >/dev/null 2>&1; then
    echo "Found Ollama: $(command -v ollama)"
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama API is responding."
        echo "Ensuring qwen3:8b is available..."
        ollama pull qwen3:8b
    else
        echo
        echo "WARNING: Ollama is installed but its API is not responding."
        echo "Try:"
        echo "  sudo systemctl enable --now ollama"
        echo "Then:"
        echo "  ollama pull qwen3:8b"
    fi
else
    echo
    echo "WARNING: Ollama is not installed."
fi

# ------------------------------------------------------------
# 11. Audio check
# ------------------------------------------------------------
echo
echo "Checking PortAudio devices..."
.venv/bin/python - <<'PY'
import sounddevice as sd
devices = sd.query_devices()
print(f"PortAudio sees {len(devices)} audio device entries.")
PY

echo
echo "============================================================"
echo "Setup complete."
echo "============================================================"
echo
echo "Project Python: $(.venv/bin/python --version)"
echo
echo 'Test Ollama + Kokoro:'
echo '  ./run.sh --text "Hello. This is a Kokoro voice test."'
echo
echo "Full voice chat:"
echo "  ./run.sh"
