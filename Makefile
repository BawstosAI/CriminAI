# AI Forensic Artist - Service Management
# Usage: make [target]

# Configuration
VENV_PATH := .venv
PYTHON := $(VENV_PATH)/bin/python
PROJECT_DIR := $(shell pwd)
DSM_DIR := $(PROJECT_DIR)/delayed-streams-modeling

# Ports
GEMINI_PORT := 8091
WS_PORT := 8000
STT_PORT := 8090

# TTS Configuration (Gradium Cloud API)
TTS_REGION := eu

# WSL Audio
export PULSE_SERVER := unix:/mnt/wslg/PulseServer

# Export port variables for services
export STT_PORT := 8090

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Default target
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)🎨 AI Forensic Artist - Service Management$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make text     # Start text-only mode"
	@echo "  make audio    # Start full voice mode (requires models)"
	@echo "  make stop     # Stop all services"
	@echo "  make status   # Check service status"

.PHONY: check-venv
check-venv: ## Check if virtual environment exists
	@if [ ! -d "$(VENV_PATH)" ]; then \
		echo "$(RED)❌ Virtual environment not found. Run: uv sync$(NC)"; \
		exit 1; \
	fi

.PHONY: check-deps
check-deps: check-venv ## Check if dependencies are installed
	@echo "$(BLUE)🔍 Checking dependencies...$(NC)"
	@$(PYTHON) -c "import google.genai; import fastapi; import websockets; print('✅ Python deps OK')" || \
		(echo "$(RED)❌ Missing dependencies. Run: uv sync$(NC)" && exit 1)
	@command -v moshi-server >/dev/null 2>&1 || \
		(echo "$(RED)❌ moshi-server not found. Install with: cargo install moshi-server --features cuda$(NC)" && exit 1)
	@echo "$(GREEN)✅ All dependencies OK$(NC)"

.PHONY: stop
stop: ## Stop all running services
	@echo "$(BLUE)🛑 Stopping all services...$(NC)"
	@-pkill -f "python src/gemini_openai_server.py" || true
	@-pkill -f "python src/websocket_server.py" || true
	@-pkill -f "moshi-server.*stt" || true
	@sleep 2
	@echo "$(GREEN)✅ All services stopped$(NC)"

.PHONY: clean
clean: stop ## Clean up processes and temporary files
	@echo "$(BLUE)🧹 Cleaning up...$(NC)"
	@rm -f *.log 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

.PHONY: status
status: ## Check status of all services
	@echo "$(BLUE)📊 Service Status:$(NC)"
	@echo ""
	@echo "$(YELLOW)Gemini API ($(GEMINI_PORT)):$(NC)"
	@curl -s http://localhost:$(GEMINI_PORT)/health 2>/dev/null && echo " ✅ Running" || echo " ❌ Not running"
	@echo ""
	@echo "$(YELLOW)WebSocket API ($(WS_PORT)):$(NC)"
	@curl -s http://localhost:$(WS_PORT)/health 2>/dev/null && echo " ✅ Running" || echo " ❌ Not running"
	@echo ""
	@echo "$(YELLOW)STT Server ($(STT_PORT)):$(NC)"
	@lsof -i:$(STT_PORT) >/dev/null 2>&1 && echo " ✅ Running" || echo " ❌ Not running"
	@echo ""
	@echo "$(YELLOW)TTS (Gradium Cloud):$(NC)"
	@[ -n "$$GRADIUM_API_KEY" ] && echo " ✅ API Key configured" || echo " ⚠️  No API key (set GRADIUM_API_KEY)"
	@echo ""

.PHONY: wait-for-service
wait-for-service: ## Internal: wait for a service to be ready
	@echo "$(BLUE)⏳ Waiting for $(SERVICE_NAME) on port $(SERVICE_PORT)...$(NC)"
	@timeout 30 bash -c 'until curl -s http://localhost:$(SERVICE_PORT)/health >/dev/null 2>&1; do sleep 1; done' || \
		(echo "$(RED)❌ $(SERVICE_NAME) failed to start$(NC)" && exit 1)
	@echo "$(GREEN)✅ $(SERVICE_NAME) is ready$(NC)"

.PHONY: start-gemini
start-gemini: check-deps ## Start Gemini API server
	@echo "$(BLUE)🧠 Starting Gemini API server...$(NC)"
	@cd $(PROJECT_DIR) && $(PYTHON) src/gemini_openai_server.py > gemini.log 2>&1 &
	@$(MAKE) wait-for-service SERVICE_NAME="Gemini API" SERVICE_PORT=$(GEMINI_PORT)

.PHONY: start-websocket
start-websocket: check-deps ## Start WebSocket server
	@echo "$(BLUE)🔌 Starting WebSocket server...$(NC)"
	@cd $(PROJECT_DIR) && $(PYTHON) src/websocket_server.py > websocket.log 2>&1 &
	@$(MAKE) wait-for-service SERVICE_NAME="WebSocket API" SERVICE_PORT=$(WS_PORT)

.PHONY: start-stt
start-stt: check-deps ## Start STT server
	@echo "$(BLUE)🎙️  Starting STT server...$(NC)"
	@if [ ! -f "$(DSM_DIR)/configs/config-stt-en-hf.toml" ]; then \
		echo "$(RED)❌ STT config not found at $(DSM_DIR)/configs/config-stt-en-hf.toml$(NC)"; \
		exit 1; \
	fi
	@cd $(DSM_DIR) && moshi-server worker --config configs/config-stt-en-hf.toml --port $(STT_PORT) > ../stt.log 2>&1 &
	@echo "$(GREEN)✅ STT server starting (models may need to download)$(NC)"



.PHONY: text
text: stop start-gemini start-websocket ## Start text-only mode (Gemini + WebSocket)
	@echo ""
	@echo "$(GREEN)🎉 Text mode ready!$(NC)"
	@echo "$(BLUE)📱 Web Interface: http://localhost:$(WS_PORT)/$(NC)"
	@echo "$(BLUE)🔗 Gemini API: http://localhost:$(GEMINI_PORT)/$(NC)"
	@echo ""
	@echo "$(YELLOW)💡 Use 'make status' to check services$(NC)"
	@echo "$(YELLOW)💡 Use 'make stop' to stop all services$(NC)"

.PHONY: audio
audio: stop start-gemini start-websocket start-stt ## Start full audio mode (all services)
	@echo ""
	@echo "$(GREEN)🎉 Audio mode started!$(NC)"
	@echo "$(BLUE)📱 Web Interface: http://localhost:$(WS_PORT)/$(NC)"
	@echo "$(BLUE)🔗 Gemini API: http://localhost:$(GEMINI_PORT)/$(NC)"
	@echo "$(BLUE)🎙️  STT Server: port $(STT_PORT)$(NC)"
	@echo "$(BLUE)🔊 TTS: Gradium Cloud API ($(TTS_REGION))$(NC)"
	@echo ""
	@echo "$(YELLOW)⚠️  Note: STT models may still be downloading$(NC)"
	@echo "$(YELLOW)💡 Ensure GRADIUM_API_KEY is set for TTS$(NC)"
	@echo "$(YELLOW)💡 Use 'make logs' to monitor download progress$(NC)"
	@echo "$(YELLOW)💡 Use 'make status' to check services$(NC)"
	@echo "$(YELLOW)💡 Use 'make stop' to stop all services$(NC)"

.PHONY: voice
voice: audio ## Alias for audio mode

.PHONY: start
start: text ## Default: start text mode

.PHONY: restart
restart: stop start ## Restart services in text mode

.PHONY: restart-audio
restart-audio: stop audio ## Restart services in audio mode

.PHONY: logs
logs: ## Show logs from all services
	@echo "$(BLUE)📋 Service Logs:$(NC)"
	@echo ""
	@if [ -f "gemini.log" ]; then \
		echo "$(YELLOW)=== Gemini API ====$(NC)"; \
		tail -10 gemini.log; echo ""; \
	fi
	@if [ -f "websocket.log" ]; then \
		echo "$(YELLOW)=== WebSocket ====$(NC)"; \
		tail -10 websocket.log; echo ""; \
	fi
	@if [ -f "stt.log" ]; then \
		echo "$(YELLOW)=== STT Server ====$(NC)"; \
		tail -10 stt.log; echo ""; \
	fi

.PHONY: test-text
test-text: ## Test text mode functionality
	@echo "$(BLUE)🧪 Testing text mode...$(NC)"
	@curl -s http://localhost:$(GEMINI_PORT)/health | grep -q "ok" && echo "$(GREEN)✅ Gemini API OK$(NC)" || echo "$(RED)❌ Gemini API failed$(NC)"
	@curl -s http://localhost:$(WS_PORT)/health | grep -q "ok" && echo "$(GREEN)✅ WebSocket API OK$(NC)" || echo "$(RED)❌ WebSocket API failed$(NC)"
	@echo "$(BLUE)🌐 Open: http://localhost:$(WS_PORT)/$(NC)"

.PHONY: open
open: ## Open web interface in browser
	@echo "$(BLUE)🌐 Opening web interface...$(NC)"
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:$(WS_PORT)/ 2>/dev/null & \
	elif command -v wsl.exe >/dev/null 2>&1; then \
		cmd.exe /c start http://localhost:$(WS_PORT)/ 2>/dev/null & \
	else \
		echo "$(YELLOW)💡 Open manually: http://localhost:$(WS_PORT)/$(NC)"; \
	fi

.PHONY: dev
dev: text open ## Start development mode (text + open browser)

# Model management
.PHONY: check-models
check-models: ## Check if voice models are downloaded
	@echo "$(BLUE)🤖 Checking voice models...$(NC)"
	@if [ -d "$$HOME/.cache/huggingface/hub" ]; then \
		echo "$(GREEN)✅ HuggingFace cache exists$(NC)"; \
		ls -la $$HOME/.cache/huggingface/hub/ | grep -E "(stt|tts)" || echo "$(YELLOW)⚠️  No voice models found in cache$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  HuggingFace cache not found$(NC)"; \
	fi

.PHONY: install-deps
install-deps: ## Install all dependencies
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	@uv sync
	@echo "$(YELLOW)⚠️  You may need to install moshi-server manually:$(NC)"
	@echo "$(YELLOW)   cargo install moshi-server --features cuda$(NC)"

# Help is the default target
.DEFAULT_GOAL := help