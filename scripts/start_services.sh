#!/bin/bash
# Start all services for the AI Forensic Artist system
# Usage: ./start_services.sh [--text-only] [--no-stt] [--no-tts]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src"

# Default ports
export GEMINI_API_PORT="${GEMINI_API_PORT:-8091}"
export STT_PORT="${STT_PORT:-8090}"
export TTS_PORT="${TTS_PORT:-8089}"
export WS_PORT="${WS_PORT:-8000}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
TEXT_ONLY=false
NO_STT=false
NO_TTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --text-only)
            TEXT_ONLY=true
            NO_STT=true
            NO_TTS=true
            shift
            ;;
        --no-stt)
            NO_STT=true
            shift
            ;;
        --no-tts)
            NO_TTS=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for required environment variables
if [[ -z "$GOOGLE_API_KEY" ]]; then
    if [[ -f "$PROJECT_DIR/.env" ]]; then
        log_info "Loading .env file..."
        export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
    fi
    
    if [[ -z "$GOOGLE_API_KEY" ]]; then
        log_error "GOOGLE_API_KEY not set. Please set it or add to .env file"
        exit 1
    fi
fi

# Cleanup function
cleanup() {
    log_info "Shutting down services..."
    
    # Kill all background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Kill by port
    for port in $GEMINI_API_PORT $WS_PORT; do
        pid=$(lsof -t -i:$port 2>/dev/null || true)
        if [[ -n "$pid" ]]; then
            kill $pid 2>/dev/null || true
        fi
    done
    
    log_success "All services stopped"
}

trap cleanup EXIT INT TERM

# Check if port is available
check_port() {
    local port=$1
    if lsof -i:$port > /dev/null 2>&1; then
        log_warn "Port $port is already in use"
        return 1
    fi
    return 0
}

# Wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    log_info "Waiting for $name to be ready..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s "$url" > /dev/null 2>&1; then
            log_success "$name is ready"
            return 0
        fi
        sleep 1
        ((attempt++))
    done
    
    log_error "$name failed to start"
    return 1
}

echo ""
echo "=========================================="
echo "   AI FORENSIC ARTIST - Service Startup"
echo "=========================================="
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [[ -d ".venv" ]]; then
    log_info "Activating virtual environment..."
    source .venv/bin/activate
fi

# Start Gemini OpenAI-compatible API server
log_info "Starting Gemini API server on port $GEMINI_API_PORT..."
check_port $GEMINI_API_PORT || exit 1
python "$SRC_DIR/gemini_openai_server.py" &
GEMINI_PID=$!

wait_for_service "http://localhost:$GEMINI_API_PORT/health" "Gemini API"

# Start STT service (if not disabled)
if [[ "$NO_STT" == "false" ]]; then
    log_info "STT service should be running on port $STT_PORT"
    log_info "Start moshi-server separately with: moshi-server --config configs/config-stt-en-hf.toml"
fi

# Start TTS service (if not disabled)
if [[ "$NO_TTS" == "false" ]]; then
    log_info "TTS service should be running on port $TTS_PORT"
    log_info "Start moshi-server separately with: moshi-server --config configs/config-tts.toml"
fi

# Start WebSocket API server
log_info "Starting WebSocket server on port $WS_PORT..."
check_port $WS_PORT || exit 1

export LLM_HOST=localhost
export LLM_PORT=$GEMINI_API_PORT
export STT_HOST=localhost
export STT_PORT=$STT_PORT
export TTS_HOST=localhost
export TTS_PORT=$TTS_PORT

python "$SRC_DIR/websocket_server.py" &
WS_PID=$!

wait_for_service "http://localhost:$WS_PORT/health" "WebSocket Server"

echo ""
echo "=========================================="
echo "   All Services Started Successfully!"
echo "=========================================="
echo ""
echo "Services running:"
echo "  • Gemini API:      http://localhost:$GEMINI_API_PORT"
echo "  • WebSocket API:   http://localhost:$WS_PORT"
echo "  • Web Interface:   http://localhost:$WS_PORT/"
echo ""

if [[ "$TEXT_ONLY" == "false" ]]; then
    echo "For voice mode, also start:"
    echo "  • STT Server (port $STT_PORT)"
    echo "  • TTS Server (port $TTS_PORT)"
    echo ""
fi

echo "Press Ctrl+C to stop all services"
echo ""

# Wait for any background process to exit
wait
