"""Real-time Speech-to-Text service wrapper using moshi-server WebSocket.

Based on kyutai-labs/unmute architecture, adapted for forensic artist system.
Includes semantic VAD (Voice Activity Detection) for turn-taking.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
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

from vad import ExponentialMovingAverage, VADConfig, ConversationState

logger = logging.getLogger(__name__)


# Constants matching unmute
FRAME_TIME_SEC = 0.08  # 80ms frames
STT_DELAY_SEC = 0.5  # Expected STT processing delay


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
    pause_probability: float = 1.0  # 1.0 = likely paused, 0.0 = speaking


class SpeechToText:
    """Real-time Speech-to-Text client using WebSocket connection to moshi-server.
    
    Includes semantic VAD for detecting when user has finished speaking.
    """

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._transcript_buffer: list[str] = []
        
        # VAD / Pause prediction (based on unmute)
        # attack = from speaking to not speaking (quick)
        # release = from not speaking to speaking (quick)
        self.pause_prediction = ExponentialMovingAverage(
            attack_time=0.01, release_time=0.01, initial_value=1.0
        )
        
        # Timing for turn-taking
        self.current_time: float = -STT_DELAY_SEC
        self.sent_samples: int = 0
        self.received_words: int = 0
        self.delay_sec: float = STT_DELAY_SEC
        
        # Callbacks
        self._on_pause_detected: Optional[Callable[[], None]] = None
        self._on_speech_started: Optional[Callable[[], None]] = None
    
    def set_turn_callbacks(
        self,
        on_pause_detected: Optional[Callable[[], None]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
    ):
        """Set callbacks for turn-taking events."""
        self._on_pause_detected = on_pause_detected
        self._on_speech_started = on_speech_started

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
                ping_interval=30,
                ping_timeout=30,
                close_timeout=10,
                max_size=10 * 1024 * 1024,  # 10MB max message size
            )
            self._running = True
            logger.info(f"Connected to STT server at {self.ws_url}")
        except Exception as e:
            logger.error(f"Failed to connect to STT server: {e}")
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
        self._ws = None
        logger.info("Disconnected from STT server")

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio chunk to STT server.
        
        Args:
            audio_data: Raw PCM audio bytes (16-bit signed, mono, 16kHz)
        """
        if not self._ws or not self._running or self._ws.closed:
            raise RuntimeError("Not connected to STT server")
        
        try:
            # Pack audio data with msgpack
            message = msgpack.packb({"audio": audio_data})
            await self._ws.send(message)
            
            # Update timing for pause detection
            num_samples = len(audio_data) // 2  # 2 bytes per int16 sample
            self.sent_samples += num_samples
            self.current_time += num_samples / self.config.sample_rate
            
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"Connection closed while sending audio: {e}")
            self._running = False
            raise RuntimeError("Connection lost") from e

    async def send_audio_array(self, audio: np.ndarray) -> None:
        """Send audio numpy array to STT server.
        
        Args:
            audio: Float32 audio array normalized to [-1, 1]
        """
        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        await self.send_audio(audio_int16.tobytes())
    
    def update_pause_prediction(self, pause_prob: float):
        """Update pause prediction from STT server response.
        
        Args:
            pause_prob: Probability that user has paused [0-1]
        """
        old_value = self.pause_prediction.value
        self.pause_prediction.update(dt=FRAME_TIME_SEC, new_value=pause_prob)
        
        # Detect transitions
        if old_value > 0.5 and self.pause_prediction.value < 0.3:
            # Speech started
            logger.debug(f"Speech started (pause: {old_value:.2f} -> {self.pause_prediction.value:.2f})")
            if self._on_speech_started:
                self._on_speech_started()
    
    def should_end_turn(self, threshold: float = 0.6) -> bool:
        """Check if we should end the user's turn based on pause prediction.
        
        Args:
            threshold: Pause probability threshold (default 0.6)
            
        Returns:
            True if user appears to have finished speaking
        """
        return self.pause_prediction.value > threshold

    async def receive_transcriptions(self) -> AsyncIterator[TranscriptionResult]:
        """Receive transcription results from STT server.
        
        Also processes pause prediction for turn-taking.
        """
        if not self._ws:
            raise RuntimeError("Not connected to STT server")
        
        # Skip first few steps as pause prediction stabilizes
        n_steps_to_wait = 12

        try:
            while self._running:
                try:
                    message = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=5.0  # Increased from 0.1s to 5s
                    )
                except asyncio.TimeoutError:
                    # Check if connection is still alive
                    if not self._ws or self._ws.closed:
                        logger.warning("STT connection lost during timeout")
                        break
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"STT connection closed: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in receive: {e}")
                    break

                # Unpack msgpack response
                try:
                    data = msgpack.unpackb(message, raw=False)
                except Exception as e:
                    logger.warning(f"Failed to unpack message: {e}")
                    continue
                
                logger.debug(f"STT message: {data}")
                
                # Handle step messages with pause prediction (like unmute's STTStepMessage)
                if "prs" in data:
                    # prs = pause prediction scores from STT
                    # prs[2] is typically the pause probability
                    self.current_time += FRAME_TIME_SEC
                    if n_steps_to_wait > 0:
                        n_steps_to_wait -= 1
                    else:
                        pause_prob = data["prs"][2] if len(data.get("prs", [])) > 2 else 0.5
                        self.update_pause_prediction(pause_prob)
                        
                        # Check for pause and trigger callback
                        if self.should_end_turn() and self._on_pause_detected:
                            self._on_pause_detected()
                
                # Handle various text field formats from different STT servers
                text_content = None
                if "text" in data:
                    text_content = data["text"]
                elif "transcript" in data:
                    text_content = data["transcript"]
                elif "result" in data and isinstance(data["result"], str):
                    text_content = data["result"]
                elif "words" in data and isinstance(data["words"], list):
                    # Handle word-by-word format
                    text_content = " ".join(w.get("text", w.get("word", "")) for w in data["words"])
                
                if text_content:
                    self.received_words += 1
                    
                    # Extract pause probability if available
                    pause_prob = self.pause_prediction.value
                    if "prs" in data and len(data["prs"]) > 2:
                        pause_prob = data["prs"][2]
                    
                    logger.info(f"STT yielding: '{text_content}'")
                    
                    yield TranscriptionResult(
                        text=text_content,
                        is_final=data.get("is_final", data.get("final", True)),
                        confidence=data.get("confidence", 1.0),
                        timestamp=data.get("timestamp", data.get("start_time", self.current_time)),
                        pause_probability=pause_prob,
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
