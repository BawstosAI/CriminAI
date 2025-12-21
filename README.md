# 🎨 AI Forensic Sketch Artist
<img width="2560" height="1399" alt="Screenshot 2025-12-18 235115" src="https://github.com/user-attachments/assets/d136163b-92b6-47e0-89f7-5038d798d190" />

CrimnAI is a bilingual (EN/FR) forensic artist agent with real-time voice and text interaction. It interviews a witness, streams speech to text, reasons with Gemini, speaks back via TTS, and auto-generates sketches. Image generation runs locally with SDXL‑Turbo and can fall back to an API when configured (demo images are used when present).

## Architecture (current stack)

```
Browser (React) ── WebSocket ── Audio/Text WS servers (8093/8092)
         │                │
         │                ├─ STT: moshi-server (asr-streaming)
         │                ├─ LLM: Gemini (OpenAI-compatible wrapper :8091)
         │                └─ TTS: moshi-server (tts_streaming)
         └─ Images: returned via WS as base64 (from image_gen)
```

## What’s included
- **Text & Audio modes** (WebSocket): audio server on `:8093`, text server on `:8092`.
- **LLM**: Gemini via OpenAI-compatible wrapper (`src/gemini_openai_server.py`, default `:8091`).
- **STT/TTS**: moshi-server endpoints (default `ws://127.0.0.1:8080/api/asr-streaming` and `/api/tts_streaming`).
- **Image gen**: `src/image_gen.py` uses SDXL-Turbo locally (GPU recommended); if an API is configured it will be used first, otherwise SDXL-Turbo runs locally. `generated/demo/` images are returned if present.
- **Language selection**: choose English or French before entering a mode; wake phrase (“Hey CrimnAI” or French variants) starts the interview in the selected language.
- **Auto image + confirmation loop**: when `[SKETCH_PROMPT: ...]` appears, the server auto-generates the image, asks for confirmation, and regenerates until satisfied.

## Quick start
```bash
# install deps
uv sync

# env (.env or export)
GOOGLE_API_KEY=...        # required for Gemini
KYUTAI_STT_URL=ws://127.0.0.1:8080
KYUTAI_TTS_URL=ws://127.0.0.1:8080
KYUTAI_API_KEY=public_token
# Optional: HF_TOKEN for HF API fallback

# Run LLM shim (default port 8091)
uv run python src/gemini_openai_server.py

# Run audio server
uv run python src/audio_websocket_server.py --port 8093

# Run text server
uv run python src/text_websocket_server.py --port 8092

# Frontend (from frontend/)
npm install
npm run dev   # open http://localhost:3000
```

### moshi-server (STT/TTS)
Ensure moshi-server is running (default `:8080`). The audio server sends an `Init` (lang=en) for compatibility; LLM responses follow your selected UI language.

### Image generation
- Local SDXL-Turbo via diffusers/torch (GPU recommended).
- HF API fallback if `HF_TOKEN` is set.
- Demo images: place a PNG/JPG in `generated/demo/` to bypass generation.

## Key files
- `src/audio_websocket_server.py`: audio WebSocket server (wake, VAD, STT/TTS, auto image, confirmation).
- `src/text_websocket_server.py`: text WebSocket server with same flow as audio.
- `src/gemini_openai_server.py`: Gemini → OpenAI-compatible `/v1/chat/completions`.
- `src/image_gen.py`: SDXL-Turbo generation with HF API fallback and demo support.
- `frontend/services/audioService.ts`: mic capture, VAD, preamp tuning.
- `frontend/services/audioPlayerService.ts`: TTS audio streaming, no overlap.

## Tips
- If STT seems deaf: ensure moshi is running, mic permissions granted, and retry normal volume (frontend preamp is on).
- If TTS overlaps: frontend resets stream per `tts_id`.
- If image gen fails: add a demo image to `generated/demo/` or set `HF_TOKEN`.

## License
MIT
