"""Real-time Text-to-Speech service using Gradium WebSocket API.

Simplified implementation using Gradium's streaming TTS WebSocket endpoint.
Documentation: https://gradium.ai/api_docs.html
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np

try:
    import websockets
except ImportError as exc:
    raise ImportError(
        "Install dependencies: pip install websockets"
    ) from exc

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech service using Gradium API."""
    api_key: str = ""  # Will be loaded from env var if not provided
    region: str = "eu"  # "eu" or "us"
    voice_id: str = "YTpq7expH9539ERJ"  # Emma (default English voice)
    model_name: str = "default"
    output_format: str = "pcm"  # "pcm" for raw audio streaming
    sample_rate: int = 48000  # Gradium PCM output is 48kHz
    
    def __post_init__(self):
        # Load API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("GRADIUM_API_KEY", "")
            if not self.api_key:
                logger.warning("No Gradium API key provided. Set GRADIUM_API_KEY env variable.")


class TextToSpeech:
    """Real-time Text-to-Speech client using Gradium WebSocket API."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False

    @property
    def ws_url(self) -> str:
        """Get the WebSocket URL based on region."""
        if self.config.region == "us":
            return "wss://us.api.gradium.ai/api/speech/tts"
        return "wss://eu.api.gradium.ai/api/speech/tts"

    async def connect(self) -> None:
        """Establish WebSocket connection to Gradium TTS server."""
        try:
            headers = {"x-api-key": self.config.api_key}
            self._ws = await websockets.connect(
                self.ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            )
            
            # Send setup message (required first message)
            setup_message = {
                "type": "setup",
                "model_name": self.config.model_name,
                "voice_id": self.config.voice_id,
                "output_format": self.config.output_format,
            }
            await self._ws.send(json.dumps(setup_message))
            
            # Wait for ready message
            ready_msg = await self._ws.recv()
            ready_data = json.loads(ready_msg)
            
            if ready_data.get("type") != "ready":
                raise RuntimeError(f"Expected 'ready' message, got: {ready_data}")
            
            self._running = True
            logger.info(f"Connected to Gradium TTS at {self.ws_url} (voice: {self.config.voice_id})")
            
        except Exception as e:
            logger.error(f"Failed to connect to Gradium TTS: {e}")
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                # Send end_of_stream message for graceful shutdown
                end_msg = {"type": "end_of_stream"}
                await self._ws.send(json.dumps(end_msg))
                await asyncio.sleep(0.1)
            except Exception:
                pass
            finally:
                await self._ws.close()
                self._ws = None
                logger.info("Disconnected from Gradium TTS")

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        """Synthesize text stream to audio stream.
        
        Args:
            text_stream: Async iterator yielding text chunks
            
        Yields:
            Audio chunks as raw PCM bytes (48kHz, 16-bit, mono)
        """
        await self.connect()
        
        try:
            # Task to send text chunks
            async def sender():
                async for text in text_stream:
                    if not self._running or not text.strip():
                        continue
                    text_message = {
                        "type": "text",
                        "text": text
                    }
                    await self._ws.send(json.dumps(text_message))
                
                # Signal end of stream
                end_msg = {"type": "end_of_stream"}
                await self._ws.send(json.dumps(end_msg))

            sender_task = asyncio.create_task(sender())
            
            # Receive audio chunks
            while self._running:
                try:
                    message = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if sender_task.done():
                        # Give time for final audio chunks
                        await asyncio.sleep(0.2)
                        try:
                            message = await asyncio.wait_for(
                                self._ws.recv(),
                                timeout=0.5
                            )
                        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                            break
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break

                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "audio":
                    # Decode base64 audio and yield as bytes
                    audio_b64 = data.get("audio", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        yield audio_bytes
                        
                elif msg_type == "end_of_stream":
                    logger.debug("Received end_of_stream from server")
                    break
                    
                elif msg_type == "error":
                    error_msg = data.get("message", "Unknown error")
                    logger.error(f"Gradium TTS error: {error_msg}")
                    break

            await sender_task
            
        finally:
            await self.disconnect()

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Complete audio as PCM bytes (48kHz, 16-bit, mono)
        """
        async def text_stream():
            yield text

        chunks = []
        async for chunk in self.synthesize_stream(text_stream()):
            chunks.append(chunk)
        
        return b"".join(chunks)



# Test client
async def main():
    """Test Gradium TTS connection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Gradium TTS client")
    parser.add_argument("--api-key", help="Gradium API key (or set GRADIUM_API_KEY env var)")
    parser.add_argument("--region", default="eu", choices=["eu", "us"], help="API region")
    parser.add_argument("--voice", default="YTpq7expH9539ERJ", help="Voice ID (default: Emma)")
    parser.add_argument("--text", default="Hello, I am your AI forensic artist. Please describe the person you saw.", help="Text to synthesize")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    config = TTSConfig(
        api_key=args.api_key or os.environ.get("GRADIUM_API_KEY", ""),
        region=args.region,
        voice_id=args.voice
    )
    
    if not config.api_key:
        print("Error: Gradium API key required. Set GRADIUM_API_KEY env var or use --api-key")
        return
    
    tts = TextToSpeech(config)

    # Test connection and synthesis
    print(f"Synthesizing: {args.text}")
    audio = await tts.synthesize(args.text)
    print(f"✅ Synthesized {len(audio)} bytes of audio (48kHz PCM)")
    
    # Save to file
    import wave
    with wave.open("tts_output.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)  # 16-bit
        f.setframerate(config.sample_rate)  # 48kHz
        f.writeframes(audio)
    print("✅ Saved to tts_output.wav")


if __name__ == "__main__":
    asyncio.run(main())
