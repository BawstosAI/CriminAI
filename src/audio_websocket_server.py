"""Audio WebSocket server for forensic artist interaction.

Handles audio streaming from frontend, transcription via Kyutai STT,
TTS responses via Kyutai TTS, and LLM responses.

Uses local moshi-server instances:
- STT: ws://localhost:8090/api/asr-streaming
- TTS: ws://localhost:8089/api/tts_streaming
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env")

try:
    import websockets
    from websockets.asyncio.server import serve, ServerConnection
except ImportError:
    raise ImportError("Install websockets: pip install websockets")

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("Install openai: pip install openai")

from kyutai_stt import KyutaiSTT, TranscriptionResult, VADResult, STT_SAMPLE_RATE
from kyutai_tts import KyutaiTTSSession, TTS_SAMPLE_RATE

# Force unbuffered output for logging
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# System prompt for forensic artist
SYSTEM_PROMPT = """You are an AI forensic sketch artist assistant. Your job is to interview a witness to gather details about a suspect's appearance and generate a detailed text prompt for image generation.

INTERVIEW GUIDELINES:
1. Ask ONE question at a time
2. Focus on specific facial features: face shape, eyes, nose, mouth, hair, distinguishing marks
3. Be conversational and supportive - witnesses may be stressed
4. After gathering sufficient details (about 8-10 questions), summarize and confirm

WHEN YOU HAVE ENOUGH DETAILS:
End your final message with a special marker containing the image prompt:
[SKETCH_PROMPT: detailed description here]

The prompt should be a single paragraph describing all gathered facial features suitable for image generation.

START by greeting the witness and asking about the overall face shape."""


@dataclass
class AudioSession:
    """Tracks a single audio user session."""
    session_id: str
    messages: list = field(default_factory=list)
    questions_asked: int = 0
    is_complete: bool = False
    sketch_prompt: Optional[str] = None
    stt: Optional[KyutaiSTT] = None
    stt_task: Optional[asyncio.Task] = None
    pending_transcription: str = ""
    tts: Optional[KyutaiTTSSession] = None
    tts_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    is_speaking: bool = False
    audio_chunks_received: int = 0


class AudioArtistServer:
    """WebSocket server for audio-based forensic artist."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8093,
        llm_host: str = "localhost",
        llm_port: int = 8091,
        stt_url: str = "ws://localhost:8090",
        tts_url: str = "ws://localhost:8089",
        tts_voice: str = "expresso_happy",
    ):
        self.host = host
        self.port = port
        self.llm_host = llm_host
        self.llm_port = llm_port
        self.stt_url = stt_url
        self.tts_url = tts_url
        self.tts_voice = tts_voice
        self.sessions: dict[str, AudioSession] = {}
        
        self.client = AsyncOpenAI(
            base_url=f"http://{llm_host}:{llm_port}/v1",
            api_key="not-needed",
        )
    
    async def handle_connection(self, websocket: ServerConnection):
        """Handle a WebSocket connection."""
        session_id = str(id(websocket))
        session = AudioSession(session_id=session_id)
        self.sessions[session_id] = session
        
        logger.info(f"New audio connection: {session_id}")
        
        try:
            # Initialize STT and TTS
            await self._init_stt(websocket, session)
            self._init_tts(session)
            
            if session.stt and session.stt.is_connected:
                # Send initial greeting (with TTS)
                greeting = await self._get_initial_greeting(session)
                await self._send_message(websocket, "assistant_message", greeting)
                await self._speak(websocket, session, greeting)
                
                # Handle incoming messages
                async for raw_message in websocket:
                    try:
                        data = json.loads(raw_message)
                        await self._handle_message(websocket, session, data)
                    except json.JSONDecodeError:
                        await self._send_error(websocket, "Invalid JSON")
                    except Exception as e:
                        logger.exception(f"Error handling message: {e}")
                        await self._send_error(websocket, str(e))
            else:
                await self._send_error(websocket, "Failed to initialize speech recognition")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {session_id}")
        finally:
            await self._cleanup_session(session)
            if session_id in self.sessions:
                del self.sessions[session_id]
    
    def _init_tts(self, session: AudioSession):
        """Initialize TTS for a session."""
        session.tts = KyutaiTTSSession(
            server_url=self.tts_url,
            voice=self.tts_voice,
        )
        logger.info(f"TTS initialized for session {session.session_id}")
    
    async def _speak(self, websocket: ServerConnection, session: AudioSession, text: str):
        """Convert text to speech and stream to client."""
        if not session.tts:
            logger.warning("No TTS session available")
            return
        
        # Use lock to prevent concurrent TTS connections (Gradium limit = 2)
        async with session.tts_lock:
            # Skip if already speaking (shouldn't happen with lock, but safety check)
            if session.is_speaking:
                logger.warning("Already speaking, skipping TTS request")
                return
            
            session.is_speaking = True
            logger.info(f"🔊 TTS speaking: {text[:80]}...")
            
            try:
                # Notify client that bot is speaking
                await self._send_message(websocket, "bot_speaking_start", None)
                
                # Stream audio chunks to client
                chunk_count = 0
                total_bytes = 0
                async for audio_chunk in session.tts.speak(text):
                    # Send audio as base64
                    audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
                    await websocket.send(json.dumps({
                        "type": "audio",
                        "audio": audio_b64,
                        "sample_rate": TTS_SAMPLE_RATE,
                        "format": "pcm_s16le",  # PCM signed 16-bit little-endian
                    }))
                    chunk_count += 1
                    total_bytes += len(audio_chunk)
                
                logger.info(f"🔊 TTS sent {chunk_count} chunks ({total_bytes} bytes)")
                
                # Notify client that bot finished speaking
                await self._send_message(websocket, "bot_speaking_end", None)
                
            except Exception as e:
                logger.exception(f"TTS error: {e}")
                try:
                    await self._send_message(websocket, "bot_speaking_end", None)
                except:
                    pass  # Connection might be closed
            finally:
                session.is_speaking = False
    
    async def _init_stt(self, websocket: ServerConnection, session: AudioSession):
        """Initialize the STT service for a session."""
        try:
            stt = KyutaiSTT(server_url=self.stt_url)
            
            # Set up callbacks
            async def on_transcription(result: TranscriptionResult):
                # Send partial transcription to frontend
                await self._send_message(
                    websocket, 
                    "transcription",
                    result.text,
                    extra={"is_partial": True}
                )
            
            async def on_turn_complete(text: str):
                # User finished speaking, process with LLM
                logger.info(f"User said: {text}")
                session.pending_transcription = text
                await self._process_user_turn(websocket, session, text)
            
            # Wrap async callbacks
            def sync_on_transcription(result):
                asyncio.create_task(on_transcription(result))
            
            def sync_on_turn_complete(text):
                asyncio.create_task(on_turn_complete(text))
            
            stt.on_transcription = sync_on_transcription
            stt.on_turn_complete = sync_on_turn_complete
            
            # Connect to Kyutai STT
            if await stt.connect():
                session.stt = stt
                session.stt_task = asyncio.create_task(stt.receive_loop())
                logger.info(f"STT initialized for session {session.session_id}")
                await self._send_message(websocket, "stt_ready", "Speech recognition ready")
            else:
                logger.error("Failed to connect to Kyutai STT")
                
        except Exception as e:
            logger.exception(f"STT initialization error: {e}")
            await self._send_error(websocket, f"STT error: {e}")
    
    async def _cleanup_session(self, session: AudioSession):
        """Clean up session resources."""
        if session.stt:
            await session.stt.disconnect()
        if session.stt_task:
            session.stt_task.cancel()
            try:
                await session.stt_task
            except asyncio.CancelledError:
                pass
    
    async def _get_initial_greeting(self, session: AudioSession) -> str:
        """Get the initial greeting from the LLM."""
        session.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        # Gemini requires at least one user message, so we add a starter
        session.messages.append({"role": "user", "content": "Start the interview."})
        
        try:
            response = await self.client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=session.messages,
                max_tokens=300,
            )
            greeting = response.choices[0].message.content
            session.messages.append({"role": "assistant", "content": greeting})
            return greeting
        except Exception as e:
            logger.error(f"LLM error: {e}")
            fallback = (
                "Hello! I'm your AI forensic artist. Let's create a composite sketch. "
                "Can you describe the overall shape of the suspect's face? "
                "Was it round, oval, square, or something else?"
            )
            session.messages.append({"role": "assistant", "content": fallback})
            return fallback
    
    async def _handle_message(
        self, 
        websocket: ServerConnection, 
        session: AudioSession, 
        data: dict
    ):
        """Process an incoming message."""
        msg_type = data.get("type")
        
        if msg_type == "audio":
            # Audio data from frontend
            audio_b64 = data.get("audio", "")
            if audio_b64 and session.stt:
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    session.audio_chunks_received += 1
                    await session.stt.send_audio(audio_bytes)
                except Exception as e:
                    logger.error(f"Audio decode error: {e}")
        
        elif msg_type == "end_turn":
            # User explicitly ended their turn (e.g., button press)
            logger.info(f"End turn received. Audio chunks: {session.audio_chunks_received}")
            
            if session.stt:
                # Notify user we're processing
                await self._send_message(websocket, "thinking", "Processing speech...")
                
                # Flush STT and wait for final transcription (handles model latency)
                text = await session.stt.flush_and_get_transcription(timeout=4.0)
                logger.info(f"Final transcription: '{text}'")
                
                if text:
                    session.stt.clear_transcription()
                    session.audio_chunks_received = 0  # Reset counter
                    await self._process_user_turn(websocket, session, text)
                else:
                    logger.warning("No transcription available after flush")
                    await self._send_message(websocket, "error", "No speech detected. Please speak and try again.")
                    # Go back to listening state
                    await self._send_message(websocket, "stt_ready", "Ready for speech")
        
        elif msg_type == "generate_image":
            # Generate the image
            prompt = data.get("prompt") or session.sketch_prompt
            await self._generate_image(websocket, session, prompt)
        
        elif msg_type == "ping":
            await self._send_message(websocket, "pong", None)
    
    async def _process_user_turn(
        self,
        websocket: ServerConnection,
        session: AudioSession,
        user_text: str
    ):
        """Process a completed user turn."""
        if not user_text.strip():
            return
        
        # Notify frontend of final transcription
        await self._send_message(
            websocket,
            "transcription",
            user_text,
            extra={"is_partial": False}
        )
        
        # Add to messages
        session.messages.append({"role": "user", "content": user_text})
        session.questions_asked += 1
        
        # Send thinking indicator
        await self._send_message(websocket, "thinking", None)
        
        # Get LLM response
        try:
            response = await self.client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=session.messages,
                max_tokens=500,
            )
            assistant_text = response.choices[0].message.content
            session.messages.append({"role": "assistant", "content": assistant_text})
            
            # Check for sketch prompt
            if "[SKETCH_PROMPT:" in assistant_text:
                start = assistant_text.find("[SKETCH_PROMPT:") + len("[SKETCH_PROMPT:")
                end = assistant_text.find("]", start)
                if end > start:
                    session.sketch_prompt = assistant_text[start:end].strip()
                    session.is_complete = True
                    
                    # Send response without the marker
                    clean_text = assistant_text[:assistant_text.find("[SKETCH_PROMPT:")].strip()
                    await self._send_message(websocket, "assistant_message", clean_text)
                    
                    # Speak the response
                    await self._speak(websocket, session, clean_text)
                    
                    # Send completion with prompt
                    await self._send_message(
                        websocket,
                        "interview_complete",
                        session.sketch_prompt,
                        extra={"questions_asked": session.questions_asked}
                    )
                    return
            
            # Send regular response
            await self._send_message(
                websocket,
                "assistant_message",
                assistant_text,
                extra={"questions_asked": session.questions_asked}
            )
            
            # Speak the response
            await self._speak(websocket, session, assistant_text)
            
        except Exception as e:
            logger.exception(f"LLM error: {e}")
            await self._send_error(websocket, f"LLM error: {e}")
    
    async def _generate_image(
        self,
        websocket: ServerConnection,
        session: AudioSession,
        prompt: Optional[str]
    ):
        """Generate the forensic sketch image."""
        if not prompt:
            await self._send_error(websocket, "No sketch prompt available")
            return
        
        await self._send_message(websocket, "generating_image", "Starting image generation...")
        
        try:
            import base64
            from image_gen import generate_image_api
            
            image_path = generate_image_api(prompt)
            
            # Read image and convert to base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            image_url = f"data:image/png;base64,{image_data}"
            
            logger.info(f"Image generated: {image_path}")
            
            await self._send_message(
                websocket,
                "image_ready",
                prompt,
                extra={"image_url": image_url, "image_path": image_path}
            )
        except Exception as e:
            logger.exception(f"Image generation error: {e}")
            await self._send_error(websocket, f"Image generation failed: {e}")
    
    async def _send_message(
        self,
        websocket: ServerConnection,
        msg_type: str,
        text: Optional[str],
        extra: Optional[dict] = None
    ):
        """Send a message to the client."""
        payload = {"type": msg_type}
        if text is not None:
            payload["text"] = text
        if extra:
            payload.update(extra)
        await websocket.send(json.dumps(payload))
    
    async def _send_error(self, websocket: ServerConnection, error: str):
        """Send an error message."""
        await self._send_message(websocket, "error", error)
    
    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"Starting Audio Artist WebSocket server on ws://{self.host}:{self.port}")
        
        async with serve(self.handle_connection, self.host, self.port):
            await asyncio.Future()  # Run forever


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Artist WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--llm-host", default=os.environ.get("LLM_HOST", "localhost"))
    parser.add_argument("--llm-port", type=int, default=int(os.environ.get("LLM_PORT", "8091")))
    parser.add_argument("--stt-url", default="ws://localhost:8090", help="Kyutai STT server URL")
    parser.add_argument("--tts-url", default="ws://localhost:8089", help="Kyutai TTS server URL")
    parser.add_argument("--voice", default="expresso_happy", help="TTS voice to use")
    args = parser.parse_args()
    
    server = AudioArtistServer(
        host=args.host,
        port=args.port,
        llm_host=args.llm_host,
        llm_port=args.llm_port,
        stt_url=args.stt_url,
        tts_url=args.tts_url,
        tts_voice=args.voice,
    )
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
