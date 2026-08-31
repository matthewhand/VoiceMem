#!/usr/bin/env bash
# VoiceMem web demo: local streaming ASR + LiteLLM chat (qwen3.8-27b) + Kokoro TTS.
# Does not use the OpenAI Realtime API.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME="${VOICEMEM_RUNTIME:-$HOME/.voicemem-runtime}"
export PATH="$RUNTIME/venv/bin:$PATH"
export VIRTUAL_ENV="$RUNTIME/venv"
export UV_CACHE_DIR="$RUNTIME/pip-cache"
export TMPDIR="$RUNTIME/tmp"
export HF_HOME="$RUNTIME/hf"
export XDG_CACHE_HOME="$RUNTIME/xdg-cache"
export VOICEMEM_MODELS_DIR="${VOICEMEM_MODELS_DIR:-$RUNTIME/models}"
export VOICEMEM_CONFIG="${VOICEMEM_CONFIG:-$ROOT/config/litellm.json}"
# APU GTT is owned by LM Studio; keep VoiceMem on CPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}"
export VOICEMEM_PORT="${VOICEMEM_PORT:-8788}"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-qwen3.8-27b}"
export OPENAI_CHAT_MODEL="${OPENAI_CHAT_MODEL:-qwen3.8-27b}"
export VOICEMEM_LANG="${VOICEMEM_LANG:-en}"
# Bilingual zipformer is better for English than the default Chinese FunASR.
export VOICEMEM_ASR="${VOICEMEM_ASR:-sherpa}"
# Default off: LMS 27B already owns ~120G GTT. Set to 1 after unloading it.
export VOICEMEM_WARMUP_AUDIO="${VOICEMEM_WARMUP_AUDIO:-0}"
cd "$ROOT"
exec python "$ROOT/web/run.py" --mode llm_tts --host 0.0.0.0 --port "$VOICEMEM_PORT" --config "$VOICEMEM_CONFIG" "$@"
