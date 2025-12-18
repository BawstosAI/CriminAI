"""WebSocket server for audio-based forensic artist interaction.

Provides real-time voice chat with the forensic artist backend using Kyutai STT/TTS.

Setup:
1. Install moshi-server (Rust):
   cargo install --features cuda moshi-server

2. Choose ONE of the following options:

   OPTION A: Run separate STT and TTS servers (recommended for development):
   
   a) Start moshi-server STT (on port 8080):
      moshi-server worker --config <path-to-config>/config-stt-en_fr-hf.toml
      (Connects to: ws://127.0.0.1:8080/api/asr-streaming)
   
   b) Start moshi-server TTS (on port 8081 to avoid conflict):
      moshi-server worker --config <path-to-config>/config-tts.toml
      # Edit config-tts.toml to set a different port, or use --port flag if supported
      # Default TTS URL: ws://127.0.0.1:8081/api/tts_streaming
   
   OPTION B: Run unified server with both STT and TTS (requires more GPU memory):
   
   A unified config file is available at:
   delayed-streams-modeling/configs/config-unified-stt-tts.toml
   
   Start the unified server:
   cd delayed-streams-modeling
   moshi-server worker --config configs/config-unified-stt-tts.toml
   
   This runs both STT and TTS on port 8080:
   - STT: ws://127.0.0.1:8080/api/asr-streaming
   - TTS: ws://127.0.0.1:8080/api/tts_streaming
   
   Then set both environment variables to the same port:
   export KYUTAI_STT_URL="ws://127.0.0.1:8080"
   export KYUTAI_TTS_URL="ws://127.0.0.1:8080"
   
   NOTE: If you get "CUDA out of memory" errors, use Option A (separate servers)
   instead, or reduce batch_size and n_q values in the config file.

3. Start Gemini LLM server (port 8091)

4. Start this audio WebSocket server:
   python src/audio_websocket_server.py --port 8093

Environment variables:
- KYUTAI_STT_URL: STT server URL (default: ws://127.0.0.1:8080)
- KYUTAI_TTS_URL: TTS server URL (default: ws://127.0.0.1:8080)
- KYUTAI_API_KEY: API key for moshi-server (default: "public_token")
- KYUTAI_TTS_VOICE: Voice to use for TTS (default: "expresso/ex03-ex01_happy_001_channel1_334s.wav")
- LLM_HOST: Gemini server host (default: localhost)
- LLM_PORT: Gemini server port (default: 8091)

IMPORTANT: If you get "Address already in use" error, make sure STT and TTS servers
are running on different ports, or use a unified server configuration.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import msgpack
import numpy as np
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
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

# Audio configuration
SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 80ms at 24kHz

# Kyutai server configuration
# Note: 
# - For separate servers: use different ports (e.g., 8080 for STT, 8081 for TTS)
# - For unified server: use the same port (e.g., 8080 for both)
#   See config-unified-stt-tts.toml for unified server setup
STT_SERVER_URL = os.environ.get("KYUTAI_STT_URL", "ws://127.0.0.1:8080")
TTS_SERVER_URL = os.environ.get("KYUTAI_TTS_URL", "ws://127.0.0.1:8080")  # Default to same port for unified server
KYUTAI_API_KEY = os.environ.get("KYUTAI_API_KEY", "public_token")
TTS_VOICE = os.environ.get("KYUTAI_TTS_VOICE", "expresso/ex03-ex01_happy_001_channel1_334s.wav")


@dataclass
class Session:
    """Tracks a single user session."""
    session_id: str
    messages: list = field(default_factory=list)
    questions_asked: int = 0
    is_complete: bool = False
    sketch_prompt: Optional[str] = None
    
    # Audio session state
    stt_ws: Optional[websockets.ClientConnection] = None
    stt_task: Optional[asyncio.Task] = None
    client_ws: Optional[ServerConnection] = None
    current_transcript: str = ""
    is_listening: bool = False
    is_speaking: bool = False
    tts_task: Optional[asyncio.Task] = None
    tts_generation: int = 0


class AudioArtistServer:
    """WebSocket server for audio-based forensic artist."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8093,
        llm_host: str = "localhost",
        llm_port: int = 8091,
    ):
        self.host = host
        self.port = port
        self.llm_host = llm_host
        self.llm_port = llm_port
        self.sessions: dict[str, Session] = {}
        
        self.client = AsyncOpenAI(
            base_url=f"http://{llm_host}:{llm_port}/v1",
            api_key="not-needed",
        )
        
    async def handle_connection(self, websocket: ServerConnection):
        """Handle a WebSocket connection."""
        session_id = str(id(websocket))
        session = Session(session_id=session_id)
        self.sessions[session_id] = session
        
        logger.info(f"New audio connection: {session_id}")
        
        try:
            # Store client websocket reference
            session.client_ws = websocket
            
            # Initialize STT connection
            await self._init_stt(session)
            
            # Get initial greeting from LLM
            greeting = await self._get_initial_greeting(session)
            
            # Send STT ready signal
            await self._send_message(websocket, "stt_ready", None)
            
            # Send initial greeting as text
            await self._send_message(websocket, "assistant_message", greeting)
            
            # Synthesize and send initial greeting audio
            # Wait a moment for STT to be ready before starting TTS
            await asyncio.sleep(0.2)
            await self._synthesize_and_send(websocket, session, greeting)
            
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
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {session_id}")
        finally:
            await self._cleanup_session(session)
            del self.sessions[session_id]
    
    async def _get_initial_greeting(self, session: Session) -> str:
        """Get the initial greeting from the LLM."""
        session.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        
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
    
    async def _init_stt(self, session: Session):
        """Initialize STT WebSocket connection to moshi-server."""
        try:
            if session.stt_task and not session.stt_task.done():
                session.stt_task.cancel()
                try:
                    await session.stt_task
                except asyncio.CancelledError:
                    pass
            if session.stt_ws:
                try:
                    await session.stt_ws.close()
                except Exception:
                    pass
            stt_url = f"{STT_SERVER_URL}/api/asr-streaming"
            headers = {"kyutai-api-key": KYUTAI_API_KEY}
            
            logger.info(f"Connecting to STT server: {stt_url}")
            session.stt_ws = await websockets.connect(stt_url, additional_headers=headers)
            
            # Start receiving transcriptions
            session.stt_task = asyncio.create_task(self._receive_stt_transcriptions(session))
            
            logger.info("STT connection established")
        except Exception as e:
            logger.error(f"Failed to connect to STT server: {e}")
            raise
    
    async def _receive_stt_transcriptions(self, session: Session):
        """Receive transcriptions from STT server."""
        try:
            async for message in session.stt_ws:
                data = msgpack.unpackb(message, raw=False)
                
                if data["type"] == "Word":
                    word = data.get("text", "")
                    if word and session.client_ws:
                        session.current_transcript += word + " "
                        # Send partial transcription to frontend
                        await self._send_message(
                            session.client_ws,
                            "transcription",
                            word,
                            extra={"is_partial": True}
                        )
                        
        except websockets.ConnectionClosed:
            logger.info("STT connection closed")
            if not session.client_ws or session.client_ws.closed:
                return
            try:
                await self._init_stt(session)
            except Exception as e:
                logger.error(f"Failed to reinitialize STT after closure: {e}")
        except Exception as e:
            logger.error(f"Error receiving STT transcriptions: {e}")

    async def _stop_tts(self, session: Session, *, send_end_event: bool = True):
        """Stop any active TTS playback task."""
        task = session.tts_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"TTS task ended with error: {e}")
        session.tts_task = None
        if send_end_event and session.is_speaking and session.client_ws:
            session.is_speaking = False
            await self._send_message(
                session.client_ws,
                "bot_speaking_end",
                None,
                extra={"tts_id": session.tts_generation},
            )

    async def _stream_tts_response(
        self,
        websocket: ServerConnection,
        session: Session,
        text: str,
        generation_id: int,
    ):
        """Open a fresh Kyutai TTS stream for this response and forward audio to the client."""
        params = {"voice": TTS_VOICE, "format": "PcmMessagePack"}
        tts_url = f"{TTS_SERVER_URL}/api/tts_streaming?{urlencode(params)}"
        headers = {"kyutai-api-key": KYUTAI_API_KEY}

        try:
            logger.info(f"[TTS:{generation_id}] Connecting to {tts_url}")
            async with websockets.connect(tts_url, additional_headers=headers) as tts_ws:
                logger.info(f"[TTS:{generation_id}] Connected")
                await self._send_message(
                    websocket,
                    "bot_speaking_start",
                    None,
                    extra={"tts_id": generation_id},
                )
                session.is_speaking = True

                eos_event = asyncio.Event()
                receiver_task = asyncio.create_task(
                    self._forward_tts_audio(
                        tts_ws,
                        websocket,
                        session,
                        generation_id,
                        eos_event,
                    )
                )

                for chunk in self._chunk_text_for_tts(text):
                    await tts_ws.send(msgpack.packb({"type": "Text", "text": chunk}))
                await tts_ws.send(msgpack.packb({"type": "Eos"}))
                eos_event.set()

                try:
                    await asyncio.wait_for(
                        receiver_task,
                        timeout=self._estimate_tts_timeout(text),
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[TTS:{generation_id}] Timeout waiting for audio stream to finish")
                    receiver_task.cancel()
                    try:
                        await receiver_task
                    except asyncio.CancelledError:
                        pass

        except asyncio.CancelledError:
            logger.info(f"[TTS:{generation_id}] Stream cancelled")
            raise
        except Exception as e:
            logger.error(f"[TTS:{generation_id}] Error during synthesis: {e}")
            await self._send_error(websocket, f"TTS error: {e}")
        finally:
            session.tts_task = None
            if session.is_speaking and session.tts_generation == generation_id:
                session.is_speaking = False
                try:
                    await self._send_message(
                        websocket,
                        "bot_speaking_end",
                        None,
                        extra={"tts_id": generation_id},
                    )
                except Exception as e:
                    logger.warning(f"[TTS:{generation_id}] Failed to send bot_speaking_end: {e}")

    async def _forward_tts_audio(
        self,
        tts_ws: websockets.ClientConnection,
        websocket: ServerConnection,
        session: Session,
        generation_id: int,
        eos_event: asyncio.Event,
    ):
        """Relay PCM audio chunks coming from the Kyutai TTS websocket to the frontend."""
        idle_timeout = 1.0

        while True:
            if session.tts_generation != generation_id:
                break

            try:
                if eos_event.is_set():
                    message_bytes = await asyncio.wait_for(tts_ws.recv(), timeout=idle_timeout)
                else:
                    message_bytes = await tts_ws.recv()
            except asyncio.TimeoutError:
                # Assume the stream is done after EOS if no audio arrives for a while
                break
            except websockets.ConnectionClosed:
                break

            msg = msgpack.unpackb(message_bytes, raw=False)
            if msg.get("type") != "Audio":
                continue

            pcm = np.array(msg["pcm"], dtype=np.float32)
            pcm_clamped = np.clip(pcm, -1.0, 1.0)
            pcm_int16 = (pcm_clamped * 32767).astype(np.int16)
            audio_base64 = base64.b64encode(pcm_int16.tobytes()).decode("utf-8")

            await self._send_message(
                websocket,
                "audio",
                None,
                extra={"audio": audio_base64, "tts_id": generation_id},
            )

        logger.info(f"[TTS:{generation_id}] Audio stream completed")

    def _chunk_text_for_tts(self, text: str, chunk_size: int = 15):
        """Yield small chunks of text to keep Kyutai TTS latency low."""
        words = text.split()
        if not words:
            return

        chunk: list[str] = []
        for word in words:
            chunk.append(word)
            if len(chunk) >= chunk_size or word.endswith((".", "!", "?", ",")):
                yield " ".join(chunk)
                chunk = []

        if chunk:
            yield " ".join(chunk)

    def _estimate_tts_timeout(self, text: str) -> float:
        """Estimate a safe timeout for TTS audio completion."""
        word_count = len(text.split())
        estimated_seconds = word_count / 2.5  # ~2.5 words/s
        return max(4.0, min(60.0, estimated_seconds + 6.0))
    
    
    async def _handle_message(
        self, 
        websocket: ServerConnection, 
        session: Session, 
        data: dict
    ):
        """Process an incoming message."""
        msg_type = data.get("type")
        
        if msg_type == "audio":
            # Audio chunk from client
            audio_base64 = data.get("audio", "")
            if not audio_base64:
                return

            # Convert base64 to float32 PCM
            try:
                audio_bytes = base64.b64decode(audio_base64)
                pcm_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                pcm_float32 = pcm_int16.astype(np.float32) / 32768.0
                
                # Send to STT server
                if session.stt_ws:
                    chunk = {"type": "Audio", "pcm": pcm_float32.tolist()}
                    msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
                    try:
                        await session.stt_ws.send(msg)
                    except websockets.ConnectionClosed:
                        logger.warning("STT websocket closed during audio send; reinitializing")
                        await self._init_stt(session)
                    except Exception as e:
                        logger.error(f"Error forwarding audio to STT: {e}")
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
        
        elif msg_type == "end_turn":
            # User finished speaking, process transcript
            if session.stt_ws:
                try:
                    await session.stt_ws.send(msgpack.packb({"type": "Eos"}))
                except Exception as e:
                    logger.error(f"Error signaling STT EOS: {e}")
            if session.current_transcript.strip():
                transcript = session.current_transcript.strip()
                session.current_transcript = ""
                
                # Send final transcription
                await self._send_message(websocket, "transcription", transcript, extra={"is_partial": False})
                
                # Add to conversation
                session.messages.append({"role": "user", "content": transcript})
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
                            
                            # Send completion with prompt
                            await self._send_message(
                                websocket, 
                                "interview_complete", 
                                session.sketch_prompt,
                                extra={"questions_asked": session.questions_asked}
                            )
                            
                            # Synthesize and send final message
                            await self._synthesize_and_send(websocket, session, clean_text)
                            return
                    
                    # Send regular response
                    await self._send_message(
                        websocket, 
                        "assistant_message", 
                        assistant_text,
                        extra={"questions_asked": session.questions_asked}
                    )
                    
                    # Synthesize and send audio
                    await self._synthesize_and_send(websocket, session, assistant_text)
                    
                except Exception as e:
                    logger.exception(f"LLM error: {e}")
                    await self._send_error(websocket, f"LLM error: {e}")
            else:
                await self._send_message(websocket, "no_transcript", None)
        
        elif msg_type == "generate_image":
            # User wants to generate the image
            prompt = data.get("prompt") or session.sketch_prompt
            if not prompt:
                await self._send_error(websocket, "No sketch prompt available")
                return
            
            await self._send_message(websocket, "generating_image", "Starting image generation...")
            
            try:
                import base64 as b64
                from image_gen import generate_image_api
                
                image_path = generate_image_api(prompt)
                
                # Read image and convert to base64 for browser display
                with open(image_path, "rb") as f:
                    image_data = b64.b64encode(f.read()).decode("utf-8")
                
                # Send as base64 data URL
                image_url = f"data:image/png;base64,{image_data}"
                
                logger.info(f"Image generated: {image_path}")
                
                # Send success with base64 URL
                await self._send_message(
                    websocket,
                    "image_ready",
                    prompt,
                    extra={"image_url": image_url, "image_path": image_path}
                )
            except Exception as e:
                logger.exception(f"Image generation error: {e}")
                await self._send_error(websocket, f"Image generation failed: {e}")
        
        elif msg_type == "ping":
            await self._send_message(websocket, "pong", None)
        
        elif msg_type == "user_interrupt":
            logger.info("User interrupt received; stopping TTS")
            await self._stop_tts(session)
    
    async def _synthesize_and_send(
        self,
        websocket: ServerConnection,
        session: Session,
        text: str
    ):
        """Synthesize text to speech and stream audio to client."""
        try:
            # Stop any existing TTS playback before starting a new utterance
            await self._stop_tts(session)

            session.tts_generation += 1
            generation_id = session.tts_generation

            # Tell frontend to flush previous audio buffers and prepare for a new stream
            await self._send_message(
                websocket,
                "stop_audio",
                None,
                extra={"tts_id": generation_id},
            )

            # Launch a background task that handles the full TTS lifecycle
            session.tts_task = asyncio.create_task(
                self._stream_tts_response(websocket, session, text, generation_id)
            )
        except Exception as e:
            logger.error(f"Error starting TTS synthesis: {e}")
            if session.client_ws:
                await self._send_message(session.client_ws, "bot_speaking_end", None)
    
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
    
    async def _cleanup_session(self, session: Session):
        """Clean up session resources."""
        # Cancel any active TTS stream
        await self._stop_tts(session)
        
        try:
            if session.stt_ws:
                await session.stt_ws.close()
        except:
            pass
        if session.stt_task and not session.stt_task.done():
            session.stt_task.cancel()
            try:
                await session.stt_task
            except asyncio.CancelledError:
                pass
    
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
    args = parser.parse_args()
    
    server = AudioArtistServer(
        host=args.host,
        port=args.port,
        llm_host=args.llm_host,
        llm_port=args.llm_port,
    )
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
