"""Real-time Speech-to-Text service wrapper using moshi-server WebSocket.

Based on kyutai-labs/unmute architecture, adapted for forensic artist system.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

import numpy as np

try:
    import msgpack
    import websockets
except ImportError as exc:
    raise ImportError(
        "Install dependencies: pip install msgpack websockets"
    ) from exc

logger = logging.getLogger(__name__)


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text service."""
    server_host: str = "localhost"
    server_port: int = 8090
    path: str = "/api/asr-streaming"
    api_key: str = "public_token"  # Default token for moshi-server
    sample_rate: int = 16000
    chunk_duration_ms: int = 80  # 80ms chunks
    language: str = "en"


@dataclass
class TranscriptionResult:
    """Result from STT transcription."""
    text: str
    is_final: bool
    confidence: float = 1.0
    timestamp: float = 0.0


class SpeechToText:
    """Real-time Speech-to-Text client using WebSocket connection to moshi-server."""

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._transcript_buffer: list[str] = []

    @property
    def ws_url(self) -> str:
        return f"ws://{self.config.server_host}:{self.config.server_port}{self.config.path}"

    async def connect(self) -> None:
        """Establish WebSocket connection to STT server."""
        try:
            headers = {"kyutai-api-key": self.config.api_key}
            self._ws = await websockets.connect(
                self.ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            )
            self._running = True
            logger.info(f"Connected to STT server at {self.ws_url}")
        except Exception as e:
            logger.error(f"Failed to connect to STT server: {e}")
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("Disconnected from STT server")

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio chunk to STT server.
        
        Args:
            audio_data: Raw PCM audio bytes (16-bit signed, mono, 16kHz)
        """
        if not self._ws or not self._running:
            raise RuntimeError("Not connected to STT server")
        
        # Pack audio data with msgpack
        message = msgpack.packb({"audio": audio_data})
        await self._ws.send(message)

    async def send_audio_array(self, audio: np.ndarray) -> None:
        """Send audio numpy array to STT server.
        
        Args:
            audio: Float32 audio array normalized to [-1, 1]
        """
        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        await self.send_audio(audio_int16.tobytes())

    async def receive_transcriptions(self) -> AsyncIterator[TranscriptionResult]:
        """Receive transcription results from STT server."""
        if not self._ws:
            raise RuntimeError("Not connected to STT server")

        try:
            while self._running:
                try:
                    message = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("STT connection closed")
                    break

                # Unpack msgpack response
                data = msgpack.unpackb(message, raw=False)
                
                if "text" in data:
                    yield TranscriptionResult(
                        text=data["text"],
                        is_final=data.get("is_final", False),
                        confidence=data.get("confidence", 1.0),
                        timestamp=data.get("timestamp", 0.0),
                    )
                elif "error" in data:
                    logger.error(f"STT error: {data['error']}")
                    
        except Exception as e:
            logger.error(f"Error receiving transcriptions: {e}")
            raise

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        on_transcript: Optional[Callable[[TranscriptionResult], None]] = None,
    ) -> str:
        """Transcribe an audio stream and return full transcript.
        
        Args:
            audio_stream: Async iterator yielding audio chunks
            on_transcript: Optional callback for intermediate results
            
        Returns:
            Complete transcription text
        """
        await self.connect()
        
        try:
            # Start receiver task
            full_transcript = []
            
            async def receiver():
                async for result in self.receive_transcriptions():
                    if on_transcript:
                        on_transcript(result)
                    if result.is_final:
                        full_transcript.append(result.text)

            receiver_task = asyncio.create_task(receiver())
            
            # Send audio
            async for chunk in audio_stream:
                await self.send_audio(chunk)
            
            # Signal end of audio
            if self._ws:
                await self._ws.send(msgpack.packb({"end": True}))
            
            # Wait for final transcriptions
            await asyncio.sleep(0.5)
            self._running = False
            await receiver_task
            
            return " ".join(full_transcript)
            
        finally:
            await self.disconnect()

    def get_chunk_samples(self) -> int:
        """Get number of samples per audio chunk."""
        return int(self.config.sample_rate * self.config.chunk_duration_ms / 1000)


class MicrophoneSTT:
    """Microphone-based Speech-to-Text using sounddevice."""

    def __init__(self, stt: SpeechToText):
        self.stt = stt
        self._running = False

    async def start(
        self,
        on_transcript: Callable[[TranscriptionResult], None],
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
    ) -> str:
        """Start microphone transcription.
        
        Args:
            on_transcript: Callback for each transcription result
            silence_threshold: RMS threshold for silence detection
            silence_duration: Seconds of silence to consider end of speech
            
        Returns:
            Final transcription when user stops speaking
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("Install sounddevice: pip install sounddevice")

        self._running = True
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        
        chunk_samples = self.stt.get_chunk_samples()
        silence_chunks = int(silence_duration * 1000 / self.stt.config.chunk_duration_ms)
        consecutive_silence = 0
        has_speech = False

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            if self._running:
                # Convert to int16 bytes
                audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
                asyncio.get_event_loop().call_soon_threadsafe(
                    audio_queue.put_nowait, audio_int16.tobytes()
                )

        async def audio_generator():
            nonlocal consecutive_silence, has_speech
            
            while self._running:
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    
                    # Check for silence
                    audio_float = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                    rms = np.sqrt(np.mean(audio_float ** 2))
                    
                    if rms > silence_threshold:
                        has_speech = True
                        consecutive_silence = 0
                    else:
                        consecutive_silence += 1
                    
                    # Stop after prolonged silence if we had speech
                    if has_speech and consecutive_silence >= silence_chunks:
                        self._running = False
                        break
                    
                    yield chunk
                    
                except asyncio.TimeoutError:
                    continue

        # Start audio stream
        stream = sd.InputStream(
            samplerate=self.stt.config.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_samples,
            callback=audio_callback,
        )

        with stream:
            logger.info("Listening... (speak now)")
            return await self.stt.transcribe_stream(audio_generator(), on_transcript)

    def stop(self):
        """Stop microphone transcription."""
        self._running = False


# Test client
async def main():
    """Test STT connection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test STT client")
    parser.add_argument("--host", default="localhost", help="STT server host")
    parser.add_argument("--port", type=int, default=8090, help="STT server port")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    config = STTConfig(server_host=args.host, server_port=args.port)
    stt = SpeechToText(config)

    def on_transcript(result: TranscriptionResult):
        prefix = "[FINAL]" if result.is_final else "[PARTIAL]"
        print(f"{prefix} {result.text}")

    if args.mic:
        mic = MicrophoneSTT(stt)
        transcript = await mic.start(on_transcript)
        print(f"\nFull transcript: {transcript}")
    else:
        # Test connection
        await stt.connect()
        print(f"Connected to {stt.ws_url}")
        await stt.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
