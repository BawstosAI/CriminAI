#!/bin/bash
# Start moshi STT server using delayed-streams-modeling
# Usage: ./start_stt.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DSM_DIR="$PROJECT_DIR/delayed-streams-modeling"

# Default port
export STT_PORT="${STT_PORT:-8090}"

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

log_info "Starting STT server on port $STT_PORT..."
log_info "Config: configs/config-stt-en-hf.toml"
echo ""

# Check if moshi-server is installed
if ! command -v moshi-server &> /dev/null; then
    log_info "Installing moshi-server..."
    cargo install moshi-server --features cuda
fi

# Start the STT server
# Note: Adjust config file path as needed for your setup
if [[ -f "configs/config-stt-en-hf.toml" ]]; then
    moshi-server worker --config configs/config-stt-en-hf.toml --port $STT_PORT
else
    log_info "Config file not found, using default settings"
    moshi-server worker --port $STT_PORT
fi
