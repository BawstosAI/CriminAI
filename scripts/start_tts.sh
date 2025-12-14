#!/bin/bash
# Start moshi TTS server using delayed-streams-modeling
# Usage: ./start_tts.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DSM_DIR="$PROJECT_DIR/delayed-streams-modeling"

# Default port
export TTS_PORT="${TTS_PORT:-8089}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Configure WSLg audio if in WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
    export PULSE_SERVER="${PULSE_SERVER:-unix:/mnt/wslg/PulseServer}"
    log_info "WSL detected, using PulseAudio at $PULSE_SERVER"
fi

cd "$DSM_DIR"

# Activate virtual environment
if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    log_info "Activating virtual environment..."
    source "$PROJECT_DIR/.venv/bin/activate"
fi

log_info "Starting TTS server on port $TTS_PORT..."
log_info "Config: configs/config-tts.toml"
echo ""

# Check if moshi-server is installed
if ! command -v moshi-server &> /dev/null; then
    log_info "Installing moshi-server..."
    cargo install moshi-server --features cuda
fi

# Start the TTS server
if [[ -f "configs/config-tts.toml" ]]; then
    moshi-server worker --config configs/config-tts.toml --port $TTS_PORT
else
    log_info "Config file not found, using default settings"
    moshi-server worker --port $TTS_PORT
fi
