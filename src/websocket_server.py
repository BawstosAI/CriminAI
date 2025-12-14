"""WebSocket API server for real-time Forensic Artist interactions.

Provides WebSocket endpoints for browser/client connections with real-time
audio streaming for STT and TTS. Implements semantic VAD for turn-taking.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
from vad import VoiceActivityDetector, VADConfig, ConversationState, calculate_audio_rms

# Configure logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Audio decoding for WebM -> PCM conversion
async def decode_webm_to_pcm(webm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Decode WebM audio to PCM using ffmpeg.
    
    Args:
        webm_data: Raw WebM audio bytes
        sample_rate: Target sample rate (default 16000)
        
    Returns:
        PCM audio bytes (16-bit signed, mono)
    """
    import tempfile
    import os
    
    logger.debug(f"decode_webm_to_pcm: input size={len(webm_data)} bytes")
    
    try:
        # Try using pydub first (if installed)
        try:
            from pydub import AudioSegment
            import io
            
            logger.debug("Trying pydub for audio decoding...")
            audio = AudioSegment.from_file(io.BytesIO(webm_data), format="webm")
            audio = audio.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
            pcm = audio.raw_data
            logger.debug(f"pydub success: output size={len(pcm)} bytes, duration={len(pcm)/2/sample_rate:.2f}s")
            return pcm
        except ImportError:
            logger.debug("pydub not available, trying ffmpeg")
            pass  # Fall through to ffmpeg
        except Exception as e:
            logger.warning(f"pydub failed: {e}, trying ffmpeg")
        
        # Write WebM to temp file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(webm_data)
            webm_path = f.name
        
        pcm_path = webm_path.replace('.webm', '.raw')
        
        # Use ffmpeg to convert (run in executor to not block)
        import asyncio
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', webm_path,
            '-f', 's16le',  # 16-bit signed little-endian
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            pcm_path
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        
        if proc.returncode != 0:
            logger.warning(f"ffmpeg error: {stderr.decode()}")
            os.unlink(webm_path)
            return b''
        
        # Read PCM output
        with open(pcm_path, 'rb') as f:
            pcm_data = f.read()
        
        # Cleanup temp files
        os.unlink(webm_path)
        os.unlink(pcm_path)
        
        return pcm_data
        
    except asyncio.TimeoutError:
        logger.warning("ffmpeg timed out")
        return b''
    except FileNotFoundError:
        logger.error("ffmpeg not found - please install ffmpeg")
        return b''
    except Exception as e:
        logger.error(f"Audio decode error: {e}")
        return b''

app = FastAPI(title="CriminAI Backend API")

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
    
    # Turn-taking events (new)
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    BOT_SPEAKING = "bot_speaking"
    BOT_FINISHED = "bot_finished"
    VAD_STATUS = "vad_status"


# Constants for turn-taking
MIN_SPEECH_DURATION_SEC = 0.3  # Minimum speech before considering a pause
PAUSE_THRESHOLD = 0.6  # Pause probability above this triggers response
SILENCE_AFTER_SPEECH_SEC = 1.5  # Silence duration to confirm end of turn
ENERGY_SILENCE_THRESHOLD = 0.02  # RMS energy threshold for silence detection
MAX_SILENCE_BEFORE_RESPONSE_SEC = 2.0  # Max silence after speech before forcing response


@dataclass
class WebSocketMessage:
    type: MessageType
    data: dict
    session_id: Optional[str] = None


@dataclass
class VoiceSessionState:
    """State for a voice session with VAD."""
    handler: ForensicArtistHandler
    vad: VoiceActivityDetector
    transcript_buffer: list[str]
    audio_buffer: list[bytes]  # Accumulate audio chunks for push-to-talk
    pcm_audio_buffer: list[bytes] = field(default_factory=list)  # Raw PCM for fallback transcription
    speech_start_time: float = 0.0
    last_speech_time: float = 0.0
    has_speech: bool = False
    is_processing: bool = False


async def fallback_transcribe(pcm_audio: bytes, sample_rate: int = 16000) -> str:
    """Fallback transcription using Whisper API or local whisper.
    
    This is called when the streaming STT doesn't provide transcripts.
    """
    import tempfile
    import os
    
    if not pcm_audio or len(pcm_audio) < 1000:
        return ""
    
    logger.info(f"🎤 Fallback transcription: {len(pcm_audio)} bytes of PCM audio")
    
    try:
        # Save PCM to WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
            # Write WAV header
            import struct
            num_samples = len(pcm_audio) // 2
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + len(pcm_audio)))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # Subchunk1 size
            f.write(struct.pack('<H', 1))   # Audio format (PCM)
            f.write(struct.pack('<H', 1))   # Num channels
            f.write(struct.pack('<I', sample_rate))  # Sample rate
            f.write(struct.pack('<I', sample_rate * 2))  # Byte rate
            f.write(struct.pack('<H', 2))   # Block align
            f.write(struct.pack('<H', 16))  # Bits per sample
            f.write(b'data')
            f.write(struct.pack('<I', len(pcm_audio)))
            f.write(pcm_audio)
        
        # Try using OpenAI Whisper API first
        try:
            from openai import AsyncOpenAI
            import os as os_module
            
            openai_key = os_module.environ.get("OPENAI_API_KEY", "")
            if openai_key:
                client = AsyncOpenAI(api_key=openai_key)
                with open(wav_path, 'rb') as audio_file:
                    transcript = await client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"
                    )
                os.unlink(wav_path)
                logger.info(f"✅ OpenAI Whisper transcription: '{transcript.text}'")
                return transcript.text
        except Exception as e:
            logger.warning(f"OpenAI Whisper failed: {e}")
        
        # Try local whisper as fallback
        try:
            proc = await asyncio.create_subprocess_exec(
                'whisper', wav_path, '--model', 'tiny', '--language', 'en', '--output_format', 'txt',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            
            txt_path = wav_path.replace('.wav', '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    text = f.read().strip()
                os.unlink(txt_path)
                os.unlink(wav_path)
                logger.info(f"✅ Local Whisper transcription: '{text}'")
                return text
        except Exception as e:
            logger.warning(f"Local Whisper failed: {e}")
        
        # Cleanup
        if os.path.exists(wav_path):
            os.unlink(wav_path)
            
    except Exception as e:
        logger.error(f"Fallback transcription error: {e}")
    
    return ""


class SessionManager:
    """Manages active WebSocket sessions."""
    
    def __init__(self):
        self.sessions: dict[str, ForensicArtistHandler] = {}
        self.voice_states: dict[str, VoiceSessionState] = {}
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
            handler = ForensicArtistHandler(self.config)
            self.sessions[session_id] = handler
            
            # Create VAD for turn-taking
            vad_config = VADConfig(
                sample_rate=16000,
                pause_threshold=PAUSE_THRESHOLD,
                min_speech_duration_sec=MIN_SPEECH_DURATION_SEC,
            )
            vad = VoiceActivityDetector(vad_config)
            
            self.voice_states[session_id] = VoiceSessionState(
                handler=handler,
                vad=vad,
                transcript_buffer=[],
                audio_buffer=[],
            )
        else:
            self.text_sessions[session_id] = TextForensicArtist(self.config)
        return session_id
    
    def get_voice_session(self, session_id: str) -> Optional[ForensicArtistHandler]:
        return self.sessions.get(session_id)
    
    def get_voice_state(self, session_id: str) -> Optional[VoiceSessionState]:
        return self.voice_states.get(session_id)
    
    def get_text_session(self, session_id: str) -> Optional[TextForensicArtist]:
        return self.text_sessions.get(session_id)
    
    async def remove_session(self, session_id: str):
        if session_id in self.sessions:
            await self.sessions[session_id].stop()
            del self.sessions[session_id]
        if session_id in self.voice_states:
            del self.voice_states[session_id]
        if session_id in self.text_sessions:
            del self.text_sessions[session_id]


session_manager = SessionManager()


async def process_turn_and_respond(
    websocket: WebSocket,
    session_id: str,
    voice_state: VoiceSessionState,
    user_text: str,
):
    """Process a complete user turn and generate bot response.
    
    This is called when VAD detects the user has finished speaking.
    """
    logger.info(f"🎯 process_turn_and_respond called with text: '{user_text}'")
    
    if not user_text.strip():
        logger.warning("⚠️ Empty user text, skipping response")
        return
    
    voice_state.is_processing = True
    logger.info(f"✅ Set is_processing=True")
    
    try:
        # Notify client we're processing
        logger.info("📤 Sending SESSION_STATE: processing")
        await websocket.send_json({
            "type": MessageType.SESSION_STATE,
            "session_id": session_id,
            "data": {"status": "processing", "user_text": user_text}
        })
        
        # Get text artist for conversation (for simplicity, use text mode logic)
        artist = session_manager.get_text_session(session_id)
        if not artist:
            logger.info(f"Creating new text session for {session_id}")
            session_manager.create_session(session_id + "_text", "text")
            artist = session_manager.text_sessions[session_id + "_text"]
        
        # Get response from LLM
        logger.info(f"🤖 Calling artist.chat() with: '{user_text}'")
        response = await artist.chat(user_text)
        logger.info(f"🤖 Got response: '{response[:100]}...' (len={len(response)})")
        
        # Notify client bot is speaking
        logger.info("📤 Sending BOT_SPEAKING")
        await websocket.send_json({
            "type": MessageType.BOT_SPEAKING,
            "session_id": session_id,
            "data": {}
        })
        
        # Send response
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
        
        # Notify client bot finished
        await websocket.send_json({
            "type": MessageType.BOT_FINISHED,
            "session_id": session_id,
            "data": {}
        })
        
        # Update VAD state
        voice_state.vad.transition_to_waiting()
        
        # Reset for next turn
        voice_state.transcript_buffer = []
        voice_state.pcm_audio_buffer = []
        voice_state.has_speech = False
        voice_state.is_processing = False
        
        # Go back to listening
        await websocket.send_json({
            "type": MessageType.SESSION_STATE,
            "session_id": session_id,
            "data": {"status": "listening"}
        })
        
    except Exception as e:
        logger.error(f"Error processing turn: {e}")
        voice_state.is_processing = False
        await websocket.send_json({
            "type": MessageType.ERROR,
            "data": {"message": f"Error generating response: {str(e)}"}
        })


@app.websocket("/v1/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time forensic artist interaction.
    
    Implements semantic VAD for turn-taking based on unmute's approach.
    """
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    mode = "text"  # Default to text mode
    stt_listener_task = None
    turn_check_task = None
    
    logger.info(f"========================================")
    logger.info(f"\ud83d\udd17 New WebSocket connection: {session_id}")
    logger.info(f"========================================")
    
    try:
        # Send initial session info
        await websocket.send_json({
            "type": MessageType.SESSION_STATE,
            "session_id": session_id,
            "data": {"status": "connected", "mode": mode}
        })
        logger.info(f"Sent initial session state to client")
        
        while True:
            # Receive message
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            msg_type = message.get("type")
            data = message.get("data", {})
            
            logger.info(f">>> Received message: type={msg_type}")
            
            if msg_type == MessageType.TEXT_INPUT:
                # Handle text input
                user_text = data.get("text", "")
                if not user_text:
                    continue
                
                # Check if we're in voice mode or text mode
                handler = session_manager.get_voice_session(session_id)
                if handler and mode == "voice":
                    # Voice mode: send greeting
                    artist_state = handler.state
                    if len(artist_state.messages) == 0:
                        greeting = "Okay, let's begin. To start, can you describe the overall shape of the suspect's face? Was it round, oval, square, or something else?"
                        await websocket.send_json({
                            "type": MessageType.ASSISTANT_TEXT,
                            "session_id": session_id,
                            "data": {"text": greeting}
                        })
                    continue
                
                # Text mode
                artist = session_manager.get_text_session(session_id)
                if not artist:
                    session_manager.create_session(session_id, "text")
                    artist = session_manager.get_text_session(session_id)
                
                # Get response
                response = await artist.chat(user_text)
                
                # Send response
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
                logger.info(">>> START_LISTENING received - switching to voice mode")
                
                # Switch to voice mode
                mode = "voice"
                session_manager.create_session(session_id, "voice")
                handler = session_manager.get_voice_session(session_id)
                voice_state = session_manager.get_voice_state(session_id)
                
                logger.info(f"Voice session created: handler={handler is not None}, voice_state={voice_state is not None}")
                
                await handler.start()
                logger.info("Handler started")
                
                # Store voice_state reference in a mutable container for closures
                state_ref = {"voice_state": voice_state, "handler": handler}
                
                # Establish STT WebSocket connection
                stt_connected = False
                try:
                    logger.info(f"Connecting to STT at {handler.stt.ws_url}...")
                    await handler.stt.connect()
                    stt_connected = True
                    logger.info(f"✅ STT connected for session {session_id}")
                    
                    # Set up turn-taking callbacks on STT
                    def on_pause_detected():
                        logger.info("STT pause detected")
                    
                    def on_speech_started():
                        if voice_state:
                            voice_state.speech_start_time = time.time()
                            voice_state.has_speech = True
                        logger.info("Speech started")
                    
                    handler.stt.set_turn_callbacks(
                        on_pause_detected=on_pause_detected,
                        on_speech_started=on_speech_started,
                    )
                    
                    # Background task to receive STT transcriptions
                    async def stt_listener():
                        try:
                            logger.info("STT listener started")
                            h = state_ref["handler"]
                            async for transcript_result in h.stt.receive_transcriptions():
                                vs = state_ref["voice_state"]
                                if not vs:
                                    break
                                
                                logger.info(f"Got transcript: '{transcript_result.text}' (final={transcript_result.is_final})")
                                
                                # Update speech timing
                                if vs and not vs.has_speech:
                                    vs.has_speech = True
                                    vs.speech_start_time = time.time()
                                    vs.vad.state.conversation_state = ConversationState.USER_SPEAKING
                                    logger.info("First speech detected via STT, starting turn")
                                    
                                    await websocket.send_json({
                                        "type": MessageType.SPEECH_STARTED,
                                        "session_id": session_id,
                                        "data": {}
                                    })
                                
                                vs.last_speech_time = time.time()
                                
                                # Add to transcript buffer
                                if transcript_result.text.strip():
                                    vs.transcript_buffer.append(transcript_result.text.strip())
                                    logger.info(f"Buffer now has {len(vs.transcript_buffer)} items: {vs.transcript_buffer}")
                                
                                # Send partial transcript
                                await websocket.send_json({
                                    "type": MessageType.TRANSCRIPT,
                                    "session_id": session_id,
                                    "data": {
                                        "text": transcript_result.text,
                                        "is_final": transcript_result.is_final,
                                        "pause_probability": transcript_result.pause_probability,
                                    }
                                })
                                
                                # Send VAD status periodically
                                await websocket.send_json({
                                    "type": MessageType.VAD_STATUS,
                                    "session_id": session_id,
                                    "data": {
                                        "pause_prediction": h.stt.pause_prediction.value,
                                        "conversation_state": vs.vad.state.conversation_state.value,
                                    }
                                })
                                
                        except Exception as e:
                            logger.error(f"STT listener error: {e}", exc_info=True)
                    
                    # Background task to check for turn completion
                    async def turn_checker():
                        """Check if user has finished speaking and trigger response."""
                        last_debug_time = 0
                        try:
                            while True:
                                await asyncio.sleep(0.1)  # Check every 100ms
                                
                                vs = state_ref["voice_state"]
                                h = state_ref["handler"]
                                
                                if not vs or vs.is_processing:
                                    continue
                                
                                current_time = time.time()
                                
                                # Debug logging every second
                                if current_time - last_debug_time > 1.0:
                                    last_debug_time = current_time
                                    pause_val = h.stt.pause_prediction.value if h else 1.0
                                    logger.info(f"Turn check: has_speech={vs.has_speech}, "
                                                f"transcripts={len(vs.transcript_buffer)}, "
                                                f"pause_pred={pause_val:.2f}, "
                                                f"silence={current_time - vs.last_speech_time:.1f}s")
                                
                                # Check if we have speech and should end turn
                                # Now also works with empty transcript buffer if we detected energy
                                if vs.has_speech and vs.last_speech_time > 0:
                                    speech_duration = current_time - vs.speech_start_time
                                    silence_duration = current_time - vs.last_speech_time
                                    pause_value = h.stt.pause_prediction.value if h else 1.0
                                    
                                    # Method 1: Semantic VAD (if STT provides pause scores)
                                    semantic_pause = pause_value > PAUSE_THRESHOLD
                                    
                                    # Method 2: Energy-based fallback (silence duration)
                                    energy_pause = silence_duration > SILENCE_AFTER_SPEECH_SEC
                                    
                                    # Method 3: Maximum silence timeout
                                    max_silence_reached = silence_duration > MAX_SILENCE_BEFORE_RESPONSE_SEC
                                    
                                    should_respond = (
                                        speech_duration > MIN_SPEECH_DURATION_SEC
                                        and (semantic_pause or energy_pause or max_silence_reached)
                                    )
                                    
                                    if should_respond:
                                        # Use transcript buffer if available, otherwise placeholder
                                        if vs.transcript_buffer:
                                            user_text = " ".join(vs.transcript_buffer)
                                        else:
                                            # Fallback: if STT didn't provide transcripts
                                            user_text = "[speech detected - transcription unavailable]"
                                            logger.warning("No transcripts available, using placeholder")
                                        
                                        reason = "semantic" if semantic_pause else ("energy" if energy_pause else "timeout")
                                        logger.info(f"Turn complete ({reason}): {user_text[:50]}... "
                                                   f"(pause={pause_value:.2f}, silence={silence_duration:.1f}s)")
                                        
                                        await websocket.send_json({
                                            "type": MessageType.SPEECH_ENDED,
                                            "session_id": session_id,
                                            "data": {"text": user_text}
                                        })
                                        
                                        # Process and respond
                                        await process_turn_and_respond(
                                            websocket, session_id, vs, user_text
                                        )
                                
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            logger.error(f"Turn checker error: {e}", exc_info=True)
                    
                    stt_listener_task = asyncio.create_task(stt_listener())
                    turn_check_task = asyncio.create_task(turn_checker())
                    
                except Exception as e:
                    logger.error(f"Failed to connect STT: {e}")
                    await websocket.send_json({
                        "type": MessageType.ERROR,
                        "data": {"message": f"Failed to connect to STT service: {str(e)}"}
                    })
                    continue
                
                await websocket.send_json({
                    "type": MessageType.SESSION_STATE,
                    "session_id": session_id,
                    "data": {"status": "listening", "mode": "voice"}
                })
            
            elif msg_type == MessageType.AUDIO_CHUNK:
                # Handle audio chunk in voice mode
                logger.debug(f">>> AUDIO_CHUNK received, data keys: {data.keys()}")
                
                handler = session_manager.get_voice_session(session_id)
                voice_state = session_manager.get_voice_state(session_id)
                
                if not handler or not voice_state:
                    logger.error(f"Voice session not found! handler={handler}, voice_state={voice_state}")
                    await websocket.send_json({
                        "type": MessageType.ERROR,
                        "data": {"message": "Voice session not started"}
                    })
                    continue
                
                # Don't process audio while bot is responding
                if voice_state.is_processing:
                    logger.debug("Skipping audio - bot is processing")
                    continue
                
                audio_b64 = data.get("audio", "")
                if not audio_b64:
                    logger.warning("Empty audio field in AUDIO_CHUNK")
                    continue
                
                logger.debug(f"Audio base64 length: {len(audio_b64)}")
                    
                try:
                    import numpy as np
                    audio_bytes = base64.b64decode(audio_b64)
                    logger.debug(f"Decoded base64 to {len(audio_bytes)} bytes")
                    
                    # Decode WebM audio to PCM using pydub/ffmpeg
                    pcm_audio = await decode_webm_to_pcm(audio_bytes)
                    
                    if pcm_audio is None or len(pcm_audio) == 0:
                        logger.warning(f"Empty audio chunk after decoding (input was {len(audio_bytes)} bytes)")
                        continue
                    
                    # Check STT connection
                    if not handler.stt._ws or handler.stt._ws.closed:
                        logger.warning(f"STT connection lost, reconnecting...")
                        await handler.stt.connect()
                    
                    logger.debug(f"PCM audio decoded: {len(pcm_audio)} bytes")
                    
                    # Store PCM for fallback transcription
                    voice_state.pcm_audio_buffer.append(pcm_audio)
                    
                    # Send PCM audio to STT
                    try:
                        await handler.stt.send_audio(pcm_audio)
                        logger.debug("Audio sent to STT successfully")
                    except Exception as stt_err:
                        logger.error(f"Failed to send audio to STT: {stt_err}")
                    
                    # Update last speech time based on audio energy
                    audio_array = np.frombuffer(pcm_audio, dtype=np.int16).astype(np.float32) / 32767.0
                    rms = calculate_audio_rms(audio_array)
                    
                    logger.debug(f"Audio RMS energy: {rms:.4f} (threshold: {ENERGY_SILENCE_THRESHOLD})")
                    
                    if rms > ENERGY_SILENCE_THRESHOLD:
                        voice_state.last_speech_time = time.time()
                        if not voice_state.has_speech:
                            voice_state.has_speech = True
                            voice_state.speech_start_time = time.time()
                            logger.info(f"🎤 SPEECH DETECTED via energy (RMS={rms:.4f})")
                        else:
                            logger.debug(f"Speech continues (RMS={rms:.4f})")
                    else:
                        if voice_state.has_speech:
                            silence_dur = time.time() - voice_state.last_speech_time
                            logger.debug(f"Silence detected (RMS={rms:.4f}), silence duration: {silence_dur:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": MessageType.ERROR,
                        "data": {"message": f"Audio streaming error: {str(e)}"}
                    })
            
            elif msg_type == MessageType.STOP_LISTENING:
                # Push-to-talk: User finished speaking, process their audio
                logger.info(">>> STOP_LISTENING received - processing turn")
                
                handler = session_manager.get_voice_session(session_id)
                voice_state = session_manager.get_voice_state(session_id)
                
                if not voice_state or voice_state.is_processing:
                    logger.warning("No voice state or already processing")
                    continue
                
                voice_state.is_processing = True
                
                # Send processing state to client
                await websocket.send_json({
                    "type": MessageType.SESSION_STATE,
                    "session_id": session_id,
                    "data": {"status": "processing"}
                })
                
                # Wait a bit for any pending audio to be processed by STT
                await asyncio.sleep(0.5)
                
                # Get transcript
                user_text = ""
                if voice_state.transcript_buffer:
                    user_text = " ".join(voice_state.transcript_buffer)
                    logger.info(f"User said (from STT): {user_text}")
                else:
                    # If no transcript but we have audio, try fallback transcription
                    if voice_state.has_speech and voice_state.pcm_audio_buffer:
                        logger.info(f"No STT transcripts, trying fallback with {len(voice_state.pcm_audio_buffer)} audio chunks")
                        combined_pcm = b''.join(voice_state.pcm_audio_buffer)
                        user_text = await fallback_transcribe(combined_pcm)
                        if user_text:
                            logger.info(f"User said (from fallback): {user_text}")
                        else:
                            logger.warning("Fallback transcription also failed")
                    elif voice_state.has_speech:
                        logger.warning("No transcripts and no PCM buffer available")
                    else:
                        logger.warning("No speech detected")
                        voice_state.is_processing = False
                        voice_state.pcm_audio_buffer = []
                        await websocket.send_json({
                            "type": MessageType.SESSION_STATE,
                            "session_id": session_id,
                            "data": {"status": "listening"}
                        })
                        continue
                
                if user_text.strip():
                    await websocket.send_json({
                        "type": MessageType.SPEECH_ENDED,
                        "session_id": session_id,
                        "data": {"text": user_text}
                    })
                    
                    await process_turn_and_respond(
                        websocket, session_id, voice_state, user_text
                    )
                else:
                    logger.warning("Empty user text after processing - no valid transcription")
                    voice_state.is_processing = False
                    voice_state.transcript_buffer = []
                    voice_state.pcm_audio_buffer = []
                    voice_state.has_speech = False
                    await websocket.send_json({
                        "type": MessageType.SESSION_STATE,
                        "session_id": session_id,
                        "data": {"status": "listening"}
                    })
                
                # Clear PCM buffer for next turn
                voice_state.pcm_audio_buffer = []
            
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
        try:
            await websocket.send_json({
                "type": MessageType.ERROR,
                "data": {"message": str(e)}
            })
        except:
            pass
    finally:
        # Cancel background tasks
        for task in [stt_listener_task, turn_check_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await session_manager.remove_session(session_id)


@app.websocket("/v1/realtime/text")
async def text_websocket_endpoint(websocket: WebSocket):
    """Text-only WebSocket endpoint."""
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    session_manager.create_session(session_id, "text")
    artist = session_manager.get_text_session(session_id)
    
    logger.info(f"New text session: {session_id}")
    
    # Send greeting
    greeting = "Hello! I'm your AI forensic artist. Let's start - can you tell me about the general build and approximate age of the person you saw?"
    artist.state.messages.append(ConversationMessage(role="assistant", content=greeting))
    
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
            
            response = await artist.chat(user_text)
            
            await websocket.send_json({
                "type": "response",
                "text": response,
                "questions_asked": artist.state.questions_asked
            })
            
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


# API Endpoints
@app.get("/")
async def root():
    """API info endpoint."""
    return {
        "message": "CriminAI Backend API",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/v1/realtime",
            "text_websocket": "/v1/realtime/text",
            "generate": "POST /api/generate",
            "health": "/health"
        },
        "frontend": "Run the React frontend separately with: cd frontend && npm run dev"
    }


class GenerateRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None


@app.post("/api/generate")
async def generate_sketch(request: GenerateRequest):
    """Generate a forensic sketch image from prompt."""
    try:
        from image_gen import generate_image
        
        output_dir = Path(__file__).parent.parent / "generated"
        output_dir.mkdir(exist_ok=True)
        
        image_id = uuid.uuid4().hex[:8]
        output_path = output_dir / f"sketch_{image_id}.png"
        
        styled_prompt = f"charcoal forensic sketch portrait, black and white, police composite drawing style, {request.prompt}"
        
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


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "sessions": len(session_manager.sessions) + len(session_manager.text_sessions)
    }


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    
    port = int(os.environ.get("WS_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
