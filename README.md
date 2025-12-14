# 🎨 AI Forensic Sketch Artist

A real-time voice-enabled AI system that interviews witnesses and generates forensic composite sketches using:

- **Gemini** as the conversational AI brain (exposed as OpenAI-compatible API)
- **moshi-server** for real-time Speech-to-Text (STT)
- **Gradium** for high-quality Text-to-Speech (TTS) via WebSocket streaming
- **SDXL-Turbo** for rapid image generation
- **WebSocket API** for browser-based real-time interaction

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Browser                               │
│                    (WebSocket Client)                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WebSocket Server                              │
│                    localhost:8000                                │
│         ┌───────────────────────────────────────────┐           │
│         │       ForensicArtistHandler               │           │
│         │  (Orchestrates STT → LLM → TTS pipeline)  │           │
│         └───────────────────────────────────────────┘           │
└─────────────┬─────────────────┬─────────────────────────────────┘
              │                 │                 │
              ▼                 ▼                 ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   STT Server    │   │  Gemini API     │   │  Gradium TTS    │
│  localhost:8090 │   │ localhost:8091  │   │   Cloud API     │
│  (moshi-server) │   │ (OpenAI-compat) │   │ (eu/us region)  │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Google Gemini  │
                    │   API Cloud     │
                    └─────────────────┘
```

## Quick Start

### 1. Environment Setup

```bash
# Clone and enter directory
cd prep_first_hackaton_ever

# Install dependencies
uv sync

# Set environment variables
export GOOGLE_API_KEY="your-gemini-api-key"
export GRADIUM_API_KEY="your-gradium-api-key"  # Required for voice mode

# Or create a .env file:
cat > .env << EOF
GOOGLE_API_KEY=your-gemini-api-key
GRADIUM_API_KEY=your-gradium-api-key
EOF
```

**Get API Keys:**
- Gemini API: https://aistudio.google.com/apikey
- Gradium API: https://gradium.ai/ (sign up for TTS service)

### 2. Start Services

**Text Mode Only (Quick Start):**
```bash
make text
```

**Full Voice Mode:**
```bash
make audio
```

**Manual Control:**
```bash
# Start individual services
make start-gemini    # Gemini LLM wrapper
make start-websocket # WebSocket server  
make start-stt       # Speech-to-text server

# Check status
make status

# Stop all services
make stop
```

### 3. Access the Interface

Open http://localhost:8000/ in your browser.

## Components

### `/src/gemini_openai_server.py`
OpenAI-compatible API wrapper for Gemini. Exposes `/v1/chat/completions` endpoint with streaming support.

### `/src/stt_service.py`
Speech-to-Text client that connects to moshi-server via WebSocket. Handles real-time audio streaming and transcription.

### `/src/tts_service.py`
Text-to-Speech client using Gradium WebSocket API. Streams high-quality natural voice responses in real-time (48kHz PCM audio).

### `/src/forensic_artist_handler.py`
Main orchestration handler that manages the interview flow:
1. Receives user speech → sends to STT
2. Transcription → sends to Gemini LLM
3. LLM response → sends to TTS
4. Returns audio to client

### `/src/websocket_server.py`
FastAPI WebSocket server providing:
- `/v1/realtime` - Full bidirectional WebSocket endpoint
- `/v1/realtime/text` - Simplified text-only endpoint
- `/` - Built-in web interface

### `/image_gen.py`
SDXL-Turbo image generation with optimized pipeline caching.

### `/faking_drawing.py`
3-stage sketch reveal animation (edges → charcoal → final).

## API Endpoints

### Gemini API (Port 8091)
```bash
# Health check
curl http://localhost:8091/health

# Chat completion (OpenAI-compatible)
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.0-flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### WebSocket API (Port 8000)
```javascript
// Connect to text endpoint
const ws = new WebSocket('ws://localhost:8000/v1/realtime/text');

// Send message
ws.send(JSON.stringify({ text: "I saw a tall man..." }));

// Receive response
ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (data.type === 'response') {
    console.log('Assistant:', data.text);
  } else if (data.type === 'sketch_ready') {
    console.log('Prompt:', data.prompt);
  }
};
```

## Configuration

Environment variables:
- `GOOGLE_API_KEY` - Google AI API key (required)
- `GEMINI_MODEL` - Model name (default: `gemini-2.0-flash`)
- `GEMINI_API_PORT` - Gemini server port (default: 8091)
- `STT_PORT` - STT server port (default: 8090)
- `TTS_PORT` - TTS server port (default: 8089)
- `WS_PORT` - WebSocket server port (default: 8000)

## Requirements

- Python 3.11+
- CUDA-capable GPU (for image generation and moshi-server)
- Rust toolchain (for moshi-server installation)
- WSLg or PulseAudio (for audio in WSL)

## License

MIT
