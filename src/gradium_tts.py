"""Gradium TTS (Text-to-Speech) service.

Connects to Gradium's WebSocket API for real-time speech synthesis.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env")

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

# Gradium TTS endpoints
GRADIUM_TTS_EU = "wss://eu.api.gradium.ai/api/speech/tts"
GRADIUM_TTS_US = "wss://us.api.gradium.ai/api/speech/tts"

# Audio format constants (output from Gradium TTS)
TTS_SAMPLE_RATE = 48000  # 48kHz
TTS_CHANNELS = 1  # Mono
TTS_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
TTS_FRAME_SIZE = 3840  # 80ms at 48kHz

# Available voices
VOICES = {
    "emma": "YTpq7expH9539ERJ",      # en-us, feminine, pleasant
    "kent": "LFZvm12tW_z0xfGo",      # en-us, masculine, relaxed
    "sydney": "jtEKaLYNn6iif5PR",    # en-us, feminine, joyful
    "john": "KWJiFWu2O9nMPYcR",      # en-us, masculine, warm broadcaster
    "eva": "ubuXFxVQwVYnZQhy",       # en-gb, feminine, british
    "jack": "m86j6D7UZpGzHsNu",      # en-gb, masculine, british
}

DEFAULT_VOICE = "john"  # Good detective voice


@dataclass
class TTSConfig:
    """Configuration for TTS."""
    voice_id: str = VOICES[DEFAULT_VOICE]
    output_format: str = "pcm"  # pcm, wav, opus
    model_name: str = "default"


class GradiumTTS:
    """Gradium Text-to-Speech client using WebSocket API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        region: str = "eu",  # "eu" or "us"
        voice: str = DEFAULT_VOICE,
    ):
        """Initialize the Gradium TTS client.
        
        Args:
            api_key: Gradium API key. If None, reads from GRADIUM_API_KEY env var.
            region: Server region - "eu" or "us"
            voice: Voice name (emma, kent, sydney, john, eva, jack) or voice_id
        """
        self.api_key = api_key or os.environ.get("GRADIUM_API_KEY")
        if not self.api_key:
            raise ValueError("GRADIUM_API_KEY not found in environment or .env file")
        
        self.endpoint = GRADIUM_TTS_EU if region == "eu" else GRADIUM_TTS_US
        
        # Resolve voice name to ID
        if voice in VOICES:
            self.voice_id = VOICES[voice]
        else:
            self.voice_id = voice  # Assume it's already a voice ID
        
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._request_id: Optional[str] = None
    
    async def connect(self) -> bool:
        """Connect to the Gradium TTS WebSocket.
        
        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to Gradium TTS: {self.endpoint}")
            
            self._ws = await connect(
                self.endpoint,
                additional_headers={"x-api-key": self.api_key},
            )
            
            # Send setup message
            setup_msg = {
                "type": "setup",
                "model_name": "default",
                "voice_id": self.voice_id,
                "output_format": "pcm",
            }
            await self._ws.send(json.dumps(setup_msg))
            logger.debug(f"Sent TTS setup: {setup_msg}")
            
            # Wait for ready message
            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            data = json.loads(response)
            
            if data.get("type") == "ready":
                self._connected = True
                self._request_id = data.get("request_id")
                logger.info(f"TTS Connected! Request ID: {self._request_id}")
                return True
            elif data.get("type") == "error":
                logger.error(f"TTS setup error: {data.get('message')}")
                return False
            else:
                logger.error(f"TTS unexpected response: {data}")
                return False
                
        except Exception as e:
            logger.exception(f"TTS connection failed: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Gradium TTS."""
        if self._ws:
            try:
                # Send end of stream
                await self._ws.send(json.dumps({"type": "end_of_stream"}))
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error during TTS disconnect: {e}")
            finally:
                self._ws = None
                self._connected = False
                self._request_id = None
        logger.info("Disconnected from Gradium TTS")
    
    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """Synthesize text to speech and yield audio chunks.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            PCM audio chunks (48kHz, 16-bit, mono)
        """
        if not self._connected or not self._ws:
            logger.error("TTS not connected")
            return
        
        try:
            # Send text message
            msg = {
                "type": "text",
                "text": text,
            }
            await self._ws.send(json.dumps(msg))
            logger.info(f"TTS sent text request: {text[:50]}...")
            
            # Receive audio chunks with timeout
            chunk_count = 0
            logger.info("TTS waiting for audio response...")
            
            # Use a timeout to avoid hanging forever
            while True:
                try:
                    # Wait for next message with 5s timeout (messages come fast during synthesis)
                    message = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"TTS timeout after {chunk_count} chunks - ending stream")
                    break
                except Exception as e:
                    logger.error(f"TTS recv error: {e}")
                    break
                
                try:
                    # Log raw message type for debugging
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "audio":
                        # Decode base64 audio
                        audio_b64 = data.get("audio", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            chunk_count += 1
                            if chunk_count == 1:
                                logger.info(f"TTS first audio chunk received ({len(audio_bytes)} bytes)")
                            elif chunk_count % 20 == 0:
                                logger.info(f"TTS received {chunk_count} audio chunks")
                            yield audio_bytes
                    
                    elif msg_type == "end_of_stream":
                        logger.info(f"TTS end of stream after {chunk_count} chunks")
                        break
                    
                    elif msg_type == "error":
                        error_msg = data.get('message', 'Unknown error')
                        error_code = data.get('code', 'N/A')
                        logger.error(f"TTS API error: {error_msg} (code: {error_code})")
                        logger.error(f"TTS full error response: {data}")
                        break
                    
                    elif msg_type == "rate_limit":
                        logger.error(f"TTS rate limit hit: {data}")
                        break
                    
                    elif msg_type == "text":
                        # Word timestamp - just ignore, don't log (too noisy)
                        pass
                    
                    else:
                        logger.warning(f"TTS unknown message type: {msg_type}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from TTS: {message[:100]}")
                    
        except Exception as e:
            logger.exception(f"TTS synthesis error: {e}")
    
    async def synthesize_full(self, text: str) -> bytes:
        """Synthesize text and return complete audio.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Complete PCM audio data
        """
        chunks = []
        async for chunk in self.synthesize(text):
            chunks.append(chunk)
        return b"".join(chunks)
    
    @property
    def is_connected(self) -> bool:
        return self._connected


class TTSSession:
    """Manages a TTS session with automatic reconnection."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        region: str = "eu",
        voice: str = DEFAULT_VOICE,
    ):
        self.api_key = api_key
        self.region = region
        self.voice = voice
        self._tts: Optional[GradiumTTS] = None
    
    async def speak(self, text: str) -> AsyncGenerator[bytes, None]:
        """Speak text, automatically connecting if needed.
        
        Args:
            text: Text to speak
            
        Yields:
            PCM audio chunks
        """
        # Create new connection for each utterance
        # (Gradium closes connection after end_of_stream)
        tts = GradiumTTS(
            api_key=self.api_key,
            region=self.region,
            voice=self.voice,
        )
        
        if await tts.connect():
            try:
                async for chunk in tts.synthesize(text):
                    yield chunk
            finally:
                await tts.disconnect()
        else:
            logger.error("Failed to connect TTS for speech")
    
    async def speak_full(self, text: str) -> bytes:
        """Speak text and return complete audio."""
        chunks = []
        async for chunk in self.speak(text):
            chunks.append(chunk)
        return b"".join(chunks)


async def test_tts():
    """Test the TTS service."""
    tts = TTSSession(region="eu", voice="john")
    
    test_text = "Hello! I am your AI forensic artist. Let me help you create a composite sketch of the suspect."
    
    print(f"Synthesizing: {test_text}")
    
    audio_data = await tts.speak_full(test_text)
    
    print(f"Generated {len(audio_data)} bytes of audio")
    print(f"Duration: {len(audio_data) / (TTS_SAMPLE_RATE * TTS_SAMPLE_WIDTH):.2f}s")
    
    # Save to file for testing
    output_file = Path(__file__).parent.parent / "generated" / "tts_test.pcm"
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_bytes(audio_data)
    print(f"Saved to: {output_file}")
    print(f"Play with: ffplay -f s16le -ar 48000 -ac 1 {output_file}")


if __name__ == "__main__":
    asyncio.run(test_tts())
