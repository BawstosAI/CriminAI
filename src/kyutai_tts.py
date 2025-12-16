"""Kyutai TTS (Text-to-Speech) service using local moshi-server.

Connects to the local Kyutai TTS Rust server via WebSocket using msgpack protocol.
Based on the delayed-streams-modeling examples.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from typing import AsyncGenerator, Optional
from urllib.parse import urlencode

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

# Audio format constants (output from Kyutai TTS)
TTS_SAMPLE_RATE = 24000  # 24kHz (moshi-server outputs 24kHz)
TTS_CHANNELS = 1  # Mono
TTS_FRAME_SIZE = 1920  # 80ms at 24kHz

# Default server settings
DEFAULT_TTS_URL = "ws://localhost:8089"
DEFAULT_API_KEY = "public_token"

# Available voices (from kyutai/tts-voices repo)
VOICES = {
    "default": "unmute-prod-website/default_voice.wav",
    "expresso_happy": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
    "expresso_sad": "expresso/ex01-ex01_sad_001_channel1_267s.wav",
    "detective": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
}

DEFAULT_VOICE = "expresso_happy"


class KyutaiTTSSession:
    """Manages a TTS session with automatic reconnection for each utterance.
    
    Creates a new WebSocket connection for each speak() call since the 
    moshi-server closes the connection after Eos.
    """
    
    def __init__(
        self,
        server_url: str = DEFAULT_TTS_URL,
        api_key: str = DEFAULT_API_KEY,
        voice: str = DEFAULT_VOICE,
    ):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        
        # Resolve voice name to path
        if voice in VOICES:
            self.voice_path = VOICES[voice]
        else:
            self.voice_path = voice
    
    async def speak(self, text: str) -> AsyncGenerator[bytes, None]:
        """Speak text, creating a new connection for each utterance.
        
        Args:
            text: Text to speak
            
        Yields:
            PCM audio chunks (24kHz, 16-bit signed int, mono)
        """
        # Build URL with voice parameter
        params = {
            "voice": self.voice_path,
            "format": "PcmMessagePack",
        }
        endpoint = f"{self.server_url}/api/tts_streaming?{urlencode(params)}"
        
        logger.info(f"Connecting to Kyutai TTS: {endpoint}")
        
        headers = {"kyutai-api-key": self.api_key}
        
        try:
            async with connect(
                endpoint,
                additional_headers=headers,
                ping_interval=None,  # Disable ping to avoid timeout issues
                ping_timeout=None,
                close_timeout=5,
            ) as ws:
                logger.info(f"Connected to Kyutai TTS, voice: {self.voice_path}")
                
                # Split text into words and send
                words = text.split()
                logger.info(f"TTS synthesizing: {text[:50]}... ({len(words)} words)")
                
                # Send all words
                for word in words:
                    msg = msgpack.packb({"type": "Text", "text": word})
                    await ws.send(msg)
                
                # Signal end of text
                await ws.send(msgpack.packb({"type": "Eos"}))
                
                # Receive and yield audio chunks
                chunk_count = 0
                async for message in ws:
                    try:
                        data = msgpack.unpackb(message, raw=False)
                        msg_type = data.get("type")
                        
                        if msg_type == "Audio":
                            pcm_float = data.get("pcm", [])
                            if pcm_float:
                                # Convert float PCM to 16-bit signed int bytes
                                pcm_int16 = [int(max(-32768, min(32767, x * 32768))) for x in pcm_float]
                                audio_bytes = struct.pack(f"<{len(pcm_int16)}h", *pcm_int16)
                                
                                chunk_count += 1
                                if chunk_count == 1:
                                    logger.info(f"TTS first audio chunk ({len(audio_bytes)} bytes)")
                                elif chunk_count % 50 == 0:
                                    logger.debug(f"TTS received {chunk_count} chunks")
                                
                                yield audio_bytes
                        
                        elif msg_type == "Eos" or msg_type is None:
                            logger.info(f"TTS complete: {chunk_count} chunks")
                            break
                    
                    except Exception as e:
                        logger.error(f"Error processing TTS message: {e}")
                        break
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"TTS connection closed: {e}")
        except Exception as e:
            logger.exception(f"TTS error: {e}")
    
    async def speak_full(self, text: str) -> bytes:
        """Speak text and return complete audio as bytes."""
        chunks = []
        async for chunk in self.speak(text):
            chunks.append(chunk)
        return b"".join(chunks)
