"""Gradium STT (Speech-to-Text) service.

Connects to Gradium's WebSocket API for real-time speech transcription.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass, field
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

# Gradium STT endpoints
GRADIUM_STT_EU = "wss://eu.api.gradium.ai/api/speech/asr"
GRADIUM_STT_US = "wss://us.api.gradium.ai/api/speech/asr"

# Audio format constants (required by Gradium)
SAMPLE_RATE = 24000  # 24kHz
CHANNELS = 1  # Mono
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
FRAME_SIZE = 1920  # 80ms at 24kHz
FRAME_DURATION_MS = 80


@dataclass
class TranscriptionResult:
    """Result from STT transcription."""
    text: str
    start_s: float
    is_final: bool = False
    

@dataclass
class VADResult:
    """Voice Activity Detection result."""
    inactivity_prob: float
    horizon_s: float
    step_idx: int
    is_turn_complete: bool = False


class GradiumSTT:
    """Gradium Speech-to-Text client using WebSocket API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        region: str = "eu",  # "eu" or "us"
        vad_threshold: float = 0.5,
    ):
        """Initialize the Gradium STT client.
        
        Args:
            api_key: Gradium API key. If None, reads from GRADIUM_API_KEY env var.
            region: Server region - "eu" or "us"
            vad_threshold: Threshold for voice activity detection (0-1)
        """
        self.api_key = api_key or os.environ.get("GRADIUM_API_KEY")
        if not self.api_key:
            raise ValueError("GRADIUM_API_KEY not found in environment or .env file")
        
        self.endpoint = GRADIUM_STT_EU if region == "eu" else GRADIUM_STT_US
        self.vad_threshold = vad_threshold
        
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._request_id: Optional[str] = None
        
        # Callbacks
        self.on_transcription: Optional[Callable[[TranscriptionResult], None]] = None
        self.on_vad: Optional[Callable[[VADResult], None]] = None
        self.on_turn_complete: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # Accumulate transcription for current turn
        self._current_transcription: list[str] = []
        
    async def connect(self) -> bool:
        """Connect to the Gradium STT WebSocket.
        
        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to Gradium STT: {self.endpoint}")
            
            self._ws = await connect(
                self.endpoint,
                additional_headers={"x-api-key": self.api_key},
            )
            
            # Send setup message
            setup_msg = {
                "type": "setup",
                "model_name": "default",
                "input_format": "pcm",
            }
            await self._ws.send(json.dumps(setup_msg))
            logger.debug(f"Sent setup: {setup_msg}")
            
            # Wait for ready message
            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            data = json.loads(response)
            
            if data.get("type") == "ready":
                self._connected = True
                self._request_id = data.get("request_id")
                logger.info(f"Connected! Request ID: {self._request_id}")
                logger.info(f"  Sample rate: {data.get('sample_rate')}Hz")
                logger.info(f"  Frame size: {data.get('frame_size')} samples")
                return True
            elif data.get("type") == "error":
                logger.error(f"Setup error: {data.get('message')}")
                return False
            else:
                logger.error(f"Unexpected response: {data}")
                return False
                
        except Exception as e:
            logger.exception(f"Connection failed: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Gradium STT."""
        if self._ws:
            try:
                # Send end of stream
                await self._ws.send(json.dumps({"type": "end_of_stream"}))
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error during disconnect: {e}")
            finally:
                self._ws = None
                self._connected = False
                self._request_id = None
                self._current_transcription = []
        logger.info("Disconnected from Gradium STT")
    
    async def send_audio(self, audio_bytes: bytes):
        """Send audio data to the STT service.
        
        Args:
            audio_bytes: Raw PCM audio data (24kHz, 16-bit, mono)
        """
        if not self._connected or not self._ws:
            logger.warning("Not connected, cannot send audio")
            return
        
        try:
            # Encode audio as base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            msg = {
                "type": "audio",
                "audio": audio_b64,
            }
            await self._ws.send(json.dumps(msg))
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    async def receive_loop(self):
        """Main loop to receive and process messages from Gradium.
        
        This should be run as a separate task.
        """
        if not self._ws:
            return
        
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message[:100]}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed: {e}")
        except Exception as e:
            logger.exception(f"Receive loop error: {e}")
        finally:
            self._connected = False
    
    async def _handle_message(self, data: dict):
        """Handle a message from Gradium."""
        msg_type = data.get("type")
        
        if msg_type == "text":
            # Transcription result
            text = data.get("text", "")
            start_s = data.get("start_s", 0.0)
            
            if text:
                self._current_transcription.append(text)
                
                result = TranscriptionResult(
                    text=text,
                    start_s=start_s,
                    is_final=False,
                )
                logger.debug(f"Transcription: '{text}' @ {start_s:.2f}s")
                
                if self.on_transcription:
                    self.on_transcription(result)
        
        elif msg_type == "step":
            # VAD (Voice Activity Detection)
            vad_data = data.get("vad", [])
            step_idx = data.get("step_idx", 0)
            
            # Use the longest horizon (index 2) for turn detection
            if len(vad_data) >= 3:
                vad_info = vad_data[2]
                inactivity_prob = vad_info.get("inactivity_prob", 0.0)
                horizon_s = vad_info.get("horizon_s", 2.0)
                
                is_turn_complete = inactivity_prob > self.vad_threshold
                
                result = VADResult(
                    inactivity_prob=inactivity_prob,
                    horizon_s=horizon_s,
                    step_idx=step_idx,
                    is_turn_complete=is_turn_complete,
                )
                
                if self.on_vad:
                    self.on_vad(result)
                
                # If turn is complete, emit the full transcription
                if is_turn_complete and self._current_transcription:
                    full_text = " ".join(self._current_transcription)
                    logger.info(f"Turn complete: '{full_text}'")
                    
                    if self.on_turn_complete:
                        self.on_turn_complete(full_text)
                    
                    # Reset for next turn
                    self._current_transcription = []
        
        elif msg_type == "end_text":
            # End of a text segment
            stop_s = data.get("stop_s", 0.0)
            logger.debug(f"End text @ {stop_s:.2f}s")
        
        elif msg_type == "end_of_stream":
            # Stream ended
            logger.info("End of stream received")
            
            # Emit any remaining transcription
            if self._current_transcription:
                full_text = " ".join(self._current_transcription)
                if self.on_turn_complete:
                    self.on_turn_complete(full_text)
                self._current_transcription = []
        
        elif msg_type == "error":
            error_msg = data.get("message", "Unknown error")
            logger.error(f"Gradium error: {error_msg}")
            if self.on_error:
                self.on_error(error_msg)
    
    def get_current_transcription(self) -> str:
        """Get the current accumulated transcription."""
        return " ".join(self._current_transcription)
    
    def clear_transcription(self):
        """Clear the current transcription buffer."""
        self._current_transcription = []
    
    @property
    def is_connected(self) -> bool:
        return self._connected


async def test_stt():
    """Test the STT service with a simple audio file."""
    import wave
    
    stt = GradiumSTT(region="eu")
    
    # Set up callbacks
    def on_transcription(result: TranscriptionResult):
        print(f"  📝 {result.text}")
    
    def on_vad(result: VADResult):
        if result.inactivity_prob > 0.3:
            print(f"  🔇 Inactivity: {result.inactivity_prob:.2f}")
    
    def on_turn_complete(text: str):
        print(f"\n✅ Turn complete: {text}\n")
    
    stt.on_transcription = on_transcription
    stt.on_vad = on_vad
    stt.on_turn_complete = on_turn_complete
    
    if not await stt.connect():
        print("Failed to connect!")
        return
    
    # Start receive loop
    receive_task = asyncio.create_task(stt.receive_loop())
    
    # If you have a test audio file, uncomment:
    # with wave.open("test.wav", "rb") as wav:
    #     while True:
    #         chunk = wav.readframes(FRAME_SIZE)
    #         if not chunk:
    #             break
    #         await stt.send_audio(chunk)
    #         await asyncio.sleep(FRAME_DURATION_MS / 1000)
    
    print("Connected! (No audio file to test)")
    print("Press Ctrl+C to exit")
    
    try:
        await asyncio.sleep(5)
    except KeyboardInterrupt:
        pass
    
    await stt.disconnect()
    receive_task.cancel()


if __name__ == "__main__":
    asyncio.run(test_stt())
