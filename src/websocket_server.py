"""WebSocket API server for real-time Forensic Artist interactions.

Provides WebSocket endpoints for browser/client connections with real-time
audio streaming for STT and TTS.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add parent directory to path for image_gen import
sys.path.insert(0, str(Path(__file__).parent.parent))

from forensic_artist_handler import (
    ForensicArtistConfig,
    ForensicArtistHandler,
    ForensicArtistState,
    TextForensicArtist,
    ConversationMessage,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Forensic Artist WebSocket API")

# CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageType(str, Enum):
    # Client -> Server
    TEXT_INPUT = "text_input"
    AUDIO_CHUNK = "audio_chunk"
    START_LISTENING = "start_listening"
    STOP_LISTENING = "stop_listening"
    END_SESSION = "end_session"
    
    # Server -> Client
    TRANSCRIPT = "transcript"
    ASSISTANT_TEXT = "assistant_text"
    ASSISTANT_AUDIO = "assistant_audio"
    SKETCH_READY = "sketch_ready"
    SESSION_STATE = "session_state"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    type: MessageType
    data: dict
    session_id: Optional[str] = None


class SessionManager:
    """Manages active WebSocket sessions."""
    
    def __init__(self):
        self.sessions: dict[str, ForensicArtistHandler] = {}
        self.text_sessions: dict[str, TextForensicArtist] = {}
        self.config = ForensicArtistConfig(
            llm_host=os.environ.get("LLM_HOST", "localhost"),
            llm_port=int(os.environ.get("LLM_PORT", "8091")),
            stt_host=os.environ.get("STT_HOST", "localhost"),
            stt_port=int(os.environ.get("STT_PORT", "8090")),
            tts_api_key=os.environ.get("GRADIUM_API_KEY", ""),
            tts_region=os.environ.get("TTS_REGION", "eu"),
            tts_voice_id=os.environ.get("TTS_VOICE_ID", "YTpq7expH9539ERJ"),
        )
    
    def create_session(self, session_id: str, mode: str = "text") -> str:
        if mode == "voice":
            self.sessions[session_id] = ForensicArtistHandler(self.config)
        else:
            self.text_sessions[session_id] = TextForensicArtist(self.config)
        return session_id
    
    def get_voice_session(self, session_id: str) -> Optional[ForensicArtistHandler]:
        return self.sessions.get(session_id)
    
    def get_text_session(self, session_id: str) -> Optional[TextForensicArtist]:
        return self.text_sessions.get(session_id)
    
    async def remove_session(self, session_id: str):
        if session_id in self.sessions:
            await self.sessions[session_id].stop()
            del self.sessions[session_id]
        if session_id in self.text_sessions:
            del self.text_sessions[session_id]


session_manager = SessionManager()


@app.websocket("/v1/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time forensic artist interaction."""
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    mode = "text"  # Default to text mode
    
    logger.info(f"New WebSocket connection: {session_id}")
    
    try:
        # Send initial session info
        await websocket.send_json({
            "type": MessageType.SESSION_STATE,
            "session_id": session_id,
            "data": {"status": "connected", "mode": mode}
        })
        
        while True:
            # Receive message
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            msg_type = message.get("type")
            data = message.get("data", {})
            
            if msg_type == MessageType.TEXT_INPUT:
                # Handle text input
                user_text = data.get("text", "")
                if not user_text:
                    continue
                
                # Get or create text session
                artist = session_manager.get_text_session(session_id)
                if not artist:
                    session_manager.create_session(session_id, "text")
                    artist = session_manager.get_text_session(session_id)
                
                # Get response (returns just the RESPONSE part)
                response = await artist.chat(user_text)
                
                # Send response with suspect profile if available
                response_data = {"text": response}
                if artist.state.suspect_profile:
                    response_data["suspect_profile"] = artist.state.suspect_profile
                
                await websocket.send_json({
                    "type": MessageType.ASSISTANT_TEXT,
                    "session_id": session_id,
                    "data": response_data
                })
                
                # Check if sketch is ready
                if artist.state.is_complete and artist.state.sketch_prompt:
                    await websocket.send_json({
                        "type": MessageType.SKETCH_READY,
                        "session_id": session_id,
                        "data": {
                            "prompt": artist.state.sketch_prompt,
                            "suspect_profile": artist.state.suspect_profile
                        }
                    })
            
            elif msg_type == MessageType.START_LISTENING:
                # Switch to voice mode
                mode = "voice"
                session_manager.create_session(session_id, "voice")
                handler = session_manager.get_voice_session(session_id)
                await handler.start()
                
                await websocket.send_json({
                    "type": MessageType.SESSION_STATE,
                    "session_id": session_id,
                    "data": {"status": "listening", "mode": "voice"}
                })
            
            elif msg_type == MessageType.AUDIO_CHUNK:
                # Handle audio chunk in voice mode
                handler = session_manager.get_voice_session(session_id)
                if not handler:
                    await websocket.send_json({
                        "type": MessageType.ERROR,
                        "data": {"message": "Voice session not started"}
                    })
                    continue
                
                # Decode base64 audio
                audio_b64 = data.get("audio", "")
                audio_bytes = base64.b64decode(audio_b64)
                
                # Send to STT
                await handler.stt.send_audio(audio_bytes)
            
            elif msg_type == MessageType.STOP_LISTENING:
                # Stop listening, process accumulated audio
                handler = session_manager.get_voice_session(session_id)
                if handler:
                    await websocket.send_json({
                        "type": MessageType.SESSION_STATE,
                        "session_id": session_id,
                        "data": {"status": "processing"}
                    })
            
            elif msg_type == MessageType.END_SESSION:
                await session_manager.remove_session(session_id)
                await websocket.send_json({
                    "type": MessageType.SESSION_STATE,
                    "session_id": session_id,
                    "data": {"status": "ended"}
                })
                break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": MessageType.ERROR,
            "data": {"message": str(e)}
        })
    finally:
        await session_manager.remove_session(session_id)


@app.websocket("/v1/realtime/text")
async def text_websocket_endpoint(websocket: WebSocket):
    """Simplified text-only WebSocket endpoint."""
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    session_manager.create_session(session_id, "text")
    artist = session_manager.get_text_session(session_id)
    
    logger.info(f"New text session: {session_id}")
    
    # Send greeting
    greeting = "Hello! I'm your AI forensic artist. Let's start - can you tell me about the general build and approximate age of the person you saw?"
    artist.state.messages.append(ConversationMessage(
        role="assistant", 
        content=greeting
    ))
    
    await websocket.send_json({
        "type": "greeting",
        "session_id": session_id,
        "text": greeting
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            user_text = data.get("text", "").strip()
            
            if not user_text:
                continue
            
            if user_text.lower() in ["quit", "exit", "bye"]:
                break
            
            # Get response
            response = await artist.chat(user_text)
            
            # Send response
            await websocket.send_json({
                "type": "response",
                "text": response,
                "questions_asked": artist.state.questions_asked
            })
            
            # Check if sketch is ready
            if artist.state.is_complete:
                await websocket.send_json({
                    "type": "sketch_ready",
                    "prompt": artist.state.sketch_prompt,
                    "conversation": [
                        {"role": m.role, "content": m.content}
                        for m in artist.state.messages
                    ]
                })
                break
    
    except WebSocketDisconnect:
        logger.info(f"Text session disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Text session error: {e}")
    finally:
        await session_manager.remove_session(session_id)


# Voice-enabled HTML client
@app.get("/voice")
async def get_voice_client():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>AI Forensic Artist - Voice Mode</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d9ff; text-align: center; }
        .chat-container {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            height: 500px;
            overflow-y: auto;
            margin-bottom: 20px;
        }
        .message {
            margin: 12px 0;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
        }
        .user { 
            background: #0f3460; 
            margin-left: auto;
            text-align: right;
        }
        .assistant { 
            background: #1a1a2e;
            border: 1px solid #00d9ff;
        }
        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        .voice-button {
            padding: 20px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            font-size: 24px;
            width: 80px;
            height: 80px;
            transition: all 0.3s ease;
        }
        .voice-button.inactive { 
            background: #16213e; 
            color: #00d9ff;
            border: 2px solid #00d9ff;
        }
        .voice-button.listening { 
            background: #ff4444; 
            color: white;
            animation: pulse 1.5s ease-in-out infinite alternate;
        }
        .voice-button:disabled { 
            background: #444; 
            cursor: not-allowed; 
            animation: none;
        }
        @keyframes pulse {
            from { transform: scale(1); }
            to { transform: scale(1.1); }
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        input {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 8px;
            background: #16213e;
            color: #eee;
            font-size: 16px;
        }
        button {
            padding: 15px 30px;
            background: #00d9ff;
            color: #1a1a2e;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            font-size: 16px;
        }
        button:hover { background: #00b8d4; }
        button:disabled { background: #444; cursor: not-allowed; }
        .status { 
            text-align: center; 
            padding: 10px; 
            color: #00d9ff;
            font-size: 14px;
        }
        .mode-toggle {
            text-align: center;
            margin-bottom: 20px;
        }
        .mode-toggle button {
            margin: 0 10px;
            padding: 10px 20px;
        }
        .mode-toggle .active {
            background: #00d9ff;
            color: #1a1a2e;
        }
        .mode-toggle .inactive {
            background: transparent;
            border: 1px solid #00d9ff;
            color: #00d9ff;
        }
    </style>
</head>
<body>
    <h1>🎙️ AI Forensic Artist - Voice Mode</h1>
    <div class="mode-toggle">
        <button class="inactive" onclick="window.location.href='/'">Text Mode</button>
        <button class="active">Voice Mode</button>
    </div>
    <div class="status" id="status">Starting continuous voice mode...</div>
    
    <div class="controls">
        <button class="voice-button listening" id="voiceBtn" onclick="toggleContinuous()">🎙️</button>
        <div style="margin-left: 20px;">
            <div style="font-size: 14px; color: #00d9ff;">Continuous Voice Mode</div>
            <div style="font-size: 12px; color: #888;">Automatically listens and responds</div>
        </div>
    </div>
    
    <div class="chat-container" id="chat"></div>
    
    <div class="input-container" style="margin-top: 10px;">
        <input type="text" id="input" placeholder="Or type a message..." style="font-size: 14px; padding: 10px;">
        <button id="send" onclick="sendText()" style="padding: 10px 20px; font-size: 14px;">Send</button>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        const status = document.getElementById('status');
        const voiceBtn = document.getElementById('voiceBtn');
        
        let ws;
        let sessionId;
        let isContinuousMode = false;
        let mediaRecorder;
        let audioStream;
        let silenceTimer;
        let isProcessing = false;
        
        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/v1/realtime`);
            
            ws.onopen = () => {
                status.textContent = 'Connected - Starting continuous voice mode...';
                input.disabled = false;
                sendBtn.disabled = false;
                voiceBtn.disabled = false;
                
                // Automatically start continuous mode
                setTimeout(() => startContinuousMode(), 1000);
            };
            
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                console.log('Received:', data);
                
                if (data.type === 'session_state') {
                    sessionId = data.session_id;
                    if (data.data.status === 'listening') {
                        isProcessing = false;
                        if (isContinuousMode) {
                            status.textContent = '🎤 Listening... (speak naturally)';
                        }
                    } else if (data.data.status === 'processing') {
                        isProcessing = true;
                        status.textContent = '🔄 Processing your speech...';
                    }
                } else if (data.type === 'transcript') {
                    addMessage(data.data.text, 'user');
                    status.textContent = '✅ You said: ' + data.data.text;
                } else if (data.type === 'assistant_text') {
                    addMessage(data.data.text, 'assistant');
                    status.textContent = '🎤 Ready for your response...';
                    
                    // After assistant responds, automatically start listening again
                    if (isContinuousMode && !isProcessing) {
                        setTimeout(() => {
                            if (!isProcessing) {
                                status.textContent = '🎤 Listening... (continue speaking)';
                            }
                        }, 2000);
                    }
                } else if (data.type === 'assistant_audio') {
                    playAudio(data.data.audio);
                } else if (data.type === 'sketch_ready') {
                    // Stop continuous mode when sketch is ready
                    stopContinuousMode();
                    status.textContent = '🎨 Interview complete! Generating sketch...';
                    setTimeout(() => generateSketchFromData(data.data), 1000);
                } else if (data.type === 'error') {
                    status.textContent = '❌ Error: ' + data.data.message;
                }
            };
            
            ws.onclose = () => {
                status.textContent = 'Disconnected. Refresh to reconnect.';
                input.disabled = true;
                sendBtn.disabled = true;
                voiceBtn.disabled = true;
            };
            
            ws.onerror = (e) => {
                console.error('WebSocket error:', e);
                status.textContent = 'Connection error';
            };
        }
        
        function addMessage(text, role) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        async function startContinuousMode() {
            try {
                audioStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 24000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                isContinuousMode = true;
                voiceBtn.className = 'voice-button listening';
                voiceBtn.textContent = '🔊';
                
                // Start listening mode
                ws.send(JSON.stringify({
                    type: 'start_listening'
                }));
                
                // Send initial greeting request
                ws.send(JSON.stringify({
                    type: 'text_input',
                    data: { text: 'Hello, I would like to start creating a forensic sketch.' }
                }));
                
                // Start continuous audio processing
                startContinuousRecording();
                
            } catch (error) {
                console.error('Error accessing microphone:', error);
                status.textContent = '❌ Microphone access denied. Please allow microphone access and refresh.';
                voiceBtn.disabled = true;
            }
        }
        
        function startContinuousRecording() {
            if (!audioStream || !isContinuousMode) return;
            
            const audioChunks = [];
            mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = async () => {
                if (audioChunks.length === 0 || !isContinuousMode) return;
                
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const arrayBuffer = await audioBlob.arrayBuffer();
                const audioData = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
                
                // Send audio data
                ws.send(JSON.stringify({
                    type: 'audio_chunk',
                    data: { audio: audioData }
                }));
                
                // Signal end of this speech segment
                ws.send(JSON.stringify({
                    type: 'stop_listening'
                }));
                
                // Wait a bit then start recording again if still in continuous mode
                if (isContinuousMode) {
                    setTimeout(() => {
                        if (isContinuousMode && !isProcessing) {
                            startContinuousRecording();
                        }
                    }, 1000);
                }
            };
            
            // Record for a reasonable chunk size (3 seconds)
            mediaRecorder.start();
            setTimeout(() => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                }
            }, 3000);
        }
        
        function stopContinuousMode() {
            isContinuousMode = false;
            
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
            
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
            }
            
            voiceBtn.className = 'voice-button inactive';
            voiceBtn.textContent = '🎤';
        }
        
        function toggleContinuous() {
            if (isContinuousMode) {
                stopContinuousMode();
                status.textContent = 'Continuous mode stopped';
            } else {
                startContinuousMode();
            }
        }
        
        function generateSketchFromData(data) {
            // Auto-generate sketch when interview is complete
            const btn = document.createElement('button');
            btn.textContent = 'Generate Sketch';
            btn.style.cssText = 'padding: 15px 30px; background: #00d9ff; color: #1a1a2e; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 16px; margin: 20px auto; display: block;';
            
            const sketchDiv = document.createElement('div');
            sketchDiv.innerHTML = `
                <div style="background: #0f3460; border: 2px solid #00d9ff; border-radius: 12px; padding: 20px; margin-top: 20px;">
                    <h3 style="color: #00d9ff; margin-top: 0;">📝 Interview Complete!</h3>
                    <p><strong>Sketch Prompt:</strong> ${data.prompt}</p>
                </div>
            `;
            sketchDiv.appendChild(btn);
            chat.appendChild(sketchDiv);
            chat.scrollTop = chat.scrollHeight;
            
            // Auto-generate after a moment
            setTimeout(() => {
                btn.disabled = true;
                btn.textContent = 'Generating...';
                generateSketch(data.prompt, btn, sketchDiv);
            }, 2000);
        }
        
        async function generateSketch(prompt, btn, container) {
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });
                const data = await response.json();
                
                if (data.success) {
                    const imgDiv = document.createElement('div');
                    imgDiv.innerHTML = `
                        <h3 style="color: #00d9ff; margin-top: 20px;">🎨 Generated Forensic Sketch</h3>
                        <img src="${data.image_url}" style="max-width: 100%; border-radius: 8px; margin-top: 10px;" />
                    `;
                    container.appendChild(imgDiv);
                    status.textContent = '✅ Sketch generated successfully!';
                    btn.remove();
                } else {
                    btn.textContent = 'Try Again';
                    btn.disabled = false;
                    status.textContent = '❌ Generation failed: ' + data.error;
                }
            } catch (e) {
                btn.textContent = 'Try Again';
                btn.disabled = false;
                status.textContent = '❌ Error: ' + e.message;
            }
        }
        
        function sendText() {
            const text = input.value.trim();
            if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            addMessage(text, 'user');
            ws.send(JSON.stringify({ 
                type: 'text_input',
                data: { text }
            }));
            input.value = '';
        }
        
        function playAudio(base64Audio) {
            const audioData = atob(base64Audio);
            const audioArray = new Uint8Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                audioArray[i] = audioData.charCodeAt(i);
            }
            const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            audio.play();
        }
        
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendText();
        });
        
        // Connect on load
        connect();
    </script>
</body>
</html>
""")

# Simple HTML client for testing
@app.get("/")
async def get_client():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>AI Forensic Artist</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d9ff; text-align: center; }
        .chat-container {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            height: 500px;
            overflow-y: auto;
            margin-bottom: 20px;
        }
        .message {
            margin: 12px 0;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
        }
        .user { 
            background: #0f3460; 
            margin-left: auto;
            text-align: right;
        }
        .assistant { 
            background: #1a1a2e;
            border: 1px solid #00d9ff;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        input {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 8px;
            background: #16213e;
            color: #eee;
            font-size: 16px;
        }
        button {
            padding: 15px 30px;
            background: #00d9ff;
            color: #1a1a2e;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            font-size: 16px;
        }
        button:hover { background: #00b8d4; }
        button:disabled { background: #444; cursor: not-allowed; }
        .sketch-prompt {
            background: #0f3460;
            border: 2px solid #00d9ff;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }
        .sketch-prompt h3 { color: #00d9ff; margin-top: 0; }
        .status { 
            text-align: center; 
            padding: 10px; 
            color: #00d9ff;
            font-size: 14px;
        }
        .mode-toggle {
            text-align: center;
            margin-bottom: 20px;
        }
        .mode-toggle button {
            margin: 0 10px;
            padding: 10px 20px;
        }
        .mode-toggle .active {
            background: #00d9ff;
            color: #1a1a2e;
        }
        .mode-toggle .inactive {
            background: transparent;
            border: 1px solid #00d9ff;
            color: #00d9ff;
        }
    </style>
</head>
<body>
    <h1>🎨 AI Forensic Sketch Artist</h1>
    <div class="mode-toggle">
        <button class="active">Text Mode</button>
        <button class="inactive" onclick="window.location.href='/voice'">Voice Mode</button>
    </div>
    <div class="status" id="status">Connecting...</div>
    <div class="chat-container" id="chat"></div>
    <div class="input-container">
        <input type="text" id="input" placeholder="Describe the person you saw..." disabled>
        <button id="send" disabled>Send</button>
    </div>
    <div class="sketch-prompt" id="sketch" style="display: none;">
        <h3>📝 Sketch Prompt Ready!</h3>
        <p id="prompt"></p>
        <button onclick="generateSketch()">Generate Sketch</button>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        const status = document.getElementById('status');
        const sketchDiv = document.getElementById('sketch');
        const promptP = document.getElementById('prompt');
        
        let ws;
        let sessionId;
        
        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/v1/realtime/text`);
            
            ws.onopen = () => {
                status.textContent = 'Connected';
                input.disabled = false;
                sendBtn.disabled = false;
            };
            
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                
                if (data.type === 'greeting') {
                    sessionId = data.session_id;
                    addMessage(data.text, 'assistant');
                } else if (data.type === 'response') {
                    addMessage(data.text, 'assistant');
                } else if (data.type === 'sketch_ready') {
                    promptP.textContent = data.prompt;
                    sketchDiv.style.display = 'block';
                    input.disabled = true;
                    sendBtn.disabled = true;
                    status.textContent = 'Interview complete!';
                }
            };
            
            ws.onclose = () => {
                status.textContent = 'Disconnected. Refresh to reconnect.';
                input.disabled = true;
                sendBtn.disabled = true;
            };
            
            ws.onerror = (e) => {
                console.error('WebSocket error:', e);
                status.textContent = 'Connection error';
            };
        }
        
        function addMessage(text, role) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        function send() {
            const text = input.value.trim();
            if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            addMessage(text, 'user');
            ws.send(JSON.stringify({ text }));
            input.value = '';
        }
        
        async function generateSketch() {
            const prompt = promptP.textContent;
            const btn = document.querySelector('.sketch-prompt button');
            btn.disabled = true;
            btn.textContent = 'Generating...';
            status.textContent = 'Generating sketch (this may take a moment)...';
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });
                const data = await response.json();
                
                if (data.success) {
                    // Show the generated image
                    const imgDiv = document.createElement('div');
                    imgDiv.innerHTML = `
                        <h3 style="color: #00d9ff; margin-top: 20px;">🎨 Generated Sketch</h3>
                        <img src="${data.image_url}" style="max-width: 100%; border-radius: 8px; margin-top: 10px;" />
                    `;
                    sketchDiv.appendChild(imgDiv);
                    status.textContent = 'Sketch generated!';
                    btn.textContent = 'Regenerate';
                    btn.disabled = false;
                } else {
                    alert('Generation failed: ' + data.error);
                    btn.textContent = 'Try Again';
                    btn.disabled = false;
                    status.textContent = 'Generation failed';
                }
            } catch (e) {
                alert('Error: ' + e.message);
                btn.textContent = 'Try Again';
                btn.disabled = false;
                status.textContent = 'Error occurred';
            }
        }
        
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') send();
        });
        sendBtn.addEventListener('click', send);
        
        connect();
    </script>
</body>
</html>
""")


# Image generation endpoint
class GenerateRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None


@app.post("/api/generate")
async def generate_sketch(request: GenerateRequest):
    """Generate a forensic sketch image from prompt."""
    try:
        from image_gen import generate_image
        
        # Create output directory if needed
        output_dir = Path(__file__).parent.parent / "generated"
        output_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        image_id = uuid.uuid4().hex[:8]
        output_path = output_dir / f"sketch_{image_id}.png"
        
        # Add charcoal sketch style to prompt
        styled_prompt = f"charcoal forensic sketch portrait, black and white, police composite drawing style, {request.prompt}"
        
        # Generate image
        logger.info(f"Generating sketch: {styled_prompt[:100]}...")
        generate_image(
            styled_prompt,
            output_path=str(output_path),
            seed=request.seed,
        )
        
        return {
            "success": True,
            "image_id": image_id,
            "image_url": f"/generated/{image_id}"
        }
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/generated/{image_id}")
async def get_generated_image(image_id: str):
    """Serve a generated image."""
    output_dir = Path(__file__).parent.parent / "generated"
    image_path = output_dir / f"sketch_{image_id}.png"
    
    if not image_path.exists():
        return {"error": "Image not found"}
    
    return FileResponse(image_path, media_type="image/png")


# Health check
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sessions": len(session_manager.sessions) + len(session_manager.text_sessions)
    }


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    
    port = int(os.environ.get("WS_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
