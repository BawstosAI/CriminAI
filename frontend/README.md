# CriminAI Frontend

A React-based frontend for the CriminAI forensic sketch artist system with full audio interaction support.

## Quick Start

1. **Start the backend first:**
   ```bash
   cd .. && make audio
   # Or manually:
   cd ../src && python websocket_server.py
   ```

2. **Install frontend dependencies:**
   ```bash
   npm install
   ```

3. **Start the frontend:**
   ```bash
   npm run dev
   ```

4. **Open in browser:** http://localhost:3000

## Features

- **Full Voice Interaction**: Speak naturally with the AI forensic artist
- **Real-time Audio Visualization**: See your voice input levels
- **Detective Avatar**: Animated character that responds during conversation
- **Automatic Sketch Generation**: Creates forensic sketch when interview completes
- **Browser TTS**: Uses browser's speech synthesis for AI responses

## Architecture

```
Frontend (React + Vite)    →    Backend (FastAPI)
     ↓                              ↓
  WebSocket (/v1/realtime)    STT Service (Port 8090)
     ↓                              ↓
  Audio Chunks            →    Speech-to-Text
     ↓                              ↓
  Transcripts             ←    Gemini LLM (Port 8091)
     ↓                              ↓
  TTS (Browser)           ←    Image Generation
```

## Environment Variables

Create a `.env` file:
```
VITE_BACKEND_HOST=localhost
VITE_BACKEND_PORT=8000
```

## Development

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
