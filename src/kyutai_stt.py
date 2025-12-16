"""Kyutai STT (Speech-to-Text) service using local moshi-server.

Connects to the local Kyutai STT Rust server via WebSocket using msgpack protocol.
Based on the delayed-streams-modeling examples.

IMPORTANT: The Kyutai STT model has inherent latency (~2-3 seconds) due to:
1. Model requires 1 second of silence at start to warm up
2. ASR delay in tokens (typically 32 tokens = ~2.5s)
3. Streaming nature of the model
"""

from __future__ import annotations

import asyncio
import logging
import struct
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import msgpack
except ImportError:
    raise ImportError("Install msgpack: pip install msgpack")

try:
    import websockets
    from websockets.asyncio.client import connect
except ImportError:
    raise ImportError("Install websockets: pip install websockets")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Audio format constants (input to Kyutai STT)
STT_SAMPLE_RATE = 24000  # 24kHz
STT_CHANNELS = 1  # Mono
STT_FRAME_SIZE = 1920  # 80ms at 24kHz

# Default server settings
DEFAULT_STT_URL = "ws://localhost:8090"
DEFAULT_API_KEY = "public_token"


@dataclass
class TranscriptionResult:
    """A transcription result with timing info."""
    text: str
    start_time: float
    stop_time: float
    is_partial: bool = False


@dataclass 
class VADResult:
    """Voice Activity Detection result."""
    is_speaking: bool
    probability: float


class KyutaiSTT:
    """Kyutai Speech-to-Text client using local moshi-server.
    
    Key behaviors:
    - Sends 1 second of silence at start to warm up the model
    - Accumulates words as they arrive
    - Has ~2-3 second latency due to model architecture
    """
    
    def __init__(
        self,
        server_url: str = DEFAULT_STT_URL,
        api_key: str = DEFAULT_API_KEY,
    ):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.endpoint = f"{self.server_url}/api/asr-streaming"
        
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None
        self._warmup_sent = False
        
        # Callbacks
        self.on_transcription: Optional[Callable[[TranscriptionResult], None]] = None
        self.on_turn_complete: Optional[Callable[[str], None]] = None
        self.on_vad: Optional[Callable[[VADResult], None]] = None
        
        # State
        self._current_transcript: list[TranscriptionResult] = []
        self._pending_words: list[TranscriptionResult] = []
        self._audio_chunks_sent = 0
        self._is_ready = False
        
        # Lock for thread-safe access to transcript
        self._transcript_lock = asyncio.Lock()
    
    async def connect(self) -> bool:
        """Connect to the Kyutai STT server."""
        try:
            logger.info(f"Connecting to Kyutai STT: {self.endpoint}")
            
            headers = {"kyutai-api-key": self.api_key}
            self._ws = await connect(
                self.endpoint,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=60,
            )
            
            self._connected = True
            logger.info(f"Connected to Kyutai STT at {self.endpoint}")
            
            return True
            
        except Exception as e:
            logger.exception(f"STT connection failed: {e}")
            self._connected = False
            return False
    
    async def send_warmup_silence(self):
        """Send 1 second of silence to warm up the model (required by Kyutai STT)."""
        if self._warmup_sent or not self._ws:
            return
            
        logger.info("Sending warmup silence to STT (1 second)...")
        silence = [0.0] * STT_SAMPLE_RATE  # 1 second of silence
        msg = msgpack.packb({"type": "Audio", "pcm": silence}, use_single_float=True)
        await self._ws.send(msg)
        self._warmup_sent = True
        logger.info("Warmup silence sent")
    
    async def disconnect(self):
        """Disconnect from the STT server."""
        self._connected = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        
        if self._ws:
            try:
                # Send end marker
                await self._ws.send(
                    msgpack.packb({"type": "Marker", "id": 0}, use_single_float=True)
                )
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error during STT disconnect: {e}")
            finally:
                self._ws = None
        
        logger.info("Disconnected from Kyutai STT")
    
    async def send_audio(self, audio_bytes: bytes):
        """Send audio data to the STT server.
        
        Args:
            audio_bytes: PCM audio data (16-bit signed int, 24kHz, mono)
        """
        if not self._connected or not self._ws:
            return
        
        # Ensure warmup silence was sent
        if not self._warmup_sent:
            await self.send_warmup_silence()
        
        try:
            # Convert bytes to float32 array
            num_samples = len(audio_bytes) // 2
            pcm_int16 = struct.unpack(f"<{num_samples}h", audio_bytes)
            pcm_float = [x / 32768.0 for x in pcm_int16]
            
            # Send as msgpack Audio message
            msg = msgpack.packb(
                {"type": "Audio", "pcm": pcm_float},
                use_single_float=True
            )
            await self._ws.send(msg)
            self._audio_chunks_sent += 1
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
    
    async def flush_and_get_transcription(self, timeout: float = 3.0) -> str:
        """Send trailing silence and wait for final transcription.
        
        This is needed because the STT model has inherent latency.
        
        Args:
            timeout: Maximum time to wait for transcription
            
        Returns:
            The accumulated transcription text
        """
        if not self._ws or not self._connected:
            return self.get_current_transcription()
        
        logger.info(f"Flushing STT with trailing silence...")
        
        # Send trailing silence to flush the model's buffer
        silence = [0.0] * STT_SAMPLE_RATE  # 1 second of silence
        for _ in range(3):  # 3 seconds total
            try:
                msg = msgpack.packb({"type": "Audio", "pcm": silence}, use_single_float=True)
                await self._ws.send(msg)
                await asyncio.sleep(0.1)  # Small delay between sends
            except Exception as e:
                logger.error(f"Error sending trailing silence: {e}")
                break
        
        # Wait for transcription to stabilize
        last_text = ""
        stable_count = 0
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            current_text = self.get_current_transcription()
            if current_text == last_text and current_text:
                stable_count += 1
                if stable_count >= 3:  # Text stable for 0.3 seconds
                    break
            else:
                stable_count = 0
                last_text = current_text
            await asyncio.sleep(0.1)
        
        result = self.get_current_transcription()
        logger.info(f"Final transcription after flush: '{result}'")
        return result
    
    async def receive_loop(self):
        """Main loop to receive and process messages from the server."""
        if not self._ws:
            return
        
        try:
            async for message in self._ws:
                if not self._connected:
                    break
                
                try:
                    data = msgpack.unpackb(message, raw=False)
                    await self._handle_message(data)
                except Exception as e:
                    logger.error(f"Error processing STT message: {e}")
                    
        except asyncio.CancelledError:
            logger.debug("STT receive loop cancelled")
        except Exception as e:
            logger.error(f"STT receive loop error: {e}")
    
    async def _handle_message(self, data: dict):
        """Handle a message from the STT server."""
        msg_type = data.get("type")
        
        if msg_type == "Ready":
            self._is_ready = True
            logger.info("STT server is ready")
            return
        
        if msg_type == "Step":
            # VAD signal - very frequent, don't log
            vad_prob = data.get("user_vad", 0.0)
            if self.on_vad:
                self.on_vad(VADResult(
                    is_speaking=vad_prob > 0.5,
                    probability=vad_prob
                ))
        
        elif msg_type == "Word":
            text = data.get("text", "")
            start_time = data.get("start_time", 0.0)
            logger.info(f"🎤 STT word: '{text}'")
            
            async with self._transcript_lock:
                word = TranscriptionResult(
                    text=text,
                    start_time=start_time,
                    stop_time=start_time,
                    is_partial=True
                )
                self._pending_words.append(word)
            
            if self.on_transcription:
                self.on_transcription(word)
        
        elif msg_type == "EndWord":
            stop_time = data.get("stop_time", 0.0)
            
            async with self._transcript_lock:
                if self._pending_words:
                    word = self._pending_words[-1]
                    word.stop_time = stop_time
                    word.is_partial = False
                    self._current_transcript.append(word)
        
        elif msg_type == "Marker":
            logger.debug("STT marker received")
        
        elif msg_type == "TurnEnd":
            full_text = self.get_current_transcription()
            logger.info(f"STT turn complete: '{full_text}'")
            
            if self.on_turn_complete and full_text:
                self.on_turn_complete(full_text)
            
            # Reset for next turn
            async with self._transcript_lock:
                self._current_transcript = []
                self._pending_words = []
    
    def get_current_transcription(self) -> str:
        """Get the current accumulated transcription."""
        words = self._current_transcript + self._pending_words
        return " ".join(w.text for w in words)
    
    def clear_transcription(self):
        """Clear the current transcription buffer."""
        self._current_transcript = []
        self._pending_words = []
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def is_ready(self) -> bool:
        return self._is_ready
