"""Voice Activity Detection (VAD) and Turn-Taking Logic.

Based on kyutai-labs/unmute semantic VAD approach that uses pause prediction
to determine when the user has finished speaking vs just pausing mid-sentence.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExponentialMovingAverage:
    """EMA that can smooth differently for attack (up) and release (down).
    
    Used for pause prediction - smooths the raw pause probability from STT
    to avoid reacting to brief fluctuations.
    
    Based on unmute/stt/exponential_moving_average.py
    """
    
    def __init__(
        self, 
        attack_time: float, 
        release_time: float, 
        initial_value: float = 0.0
    ):
        """Initialize EMA.
        
        Args:
            attack_time: Time in seconds to reach 50% of target when value increases.
            release_time: Time in seconds to decay to 50% of target when value decreases.
            initial_value: Initial value of the EMA.
        """
        self.attack_time = attack_time
        self.release_time = release_time
        self.value = initial_value
    
    def update(self, *, dt: float, new_value: float) -> float:
        """Update the EMA with a new value.
        
        Args:
            dt: Time delta in seconds since last update.
            new_value: The new raw value to incorporate.
            
        Returns:
            Updated smoothed value.
        """
        assert dt > 0.0, f"dt must be positive, got {dt=}"
        assert new_value >= 0.0, f"new_value must be non-negative, got {new_value=}"
        
        if new_value > self.value:
            # Attack: value is increasing (user stopping speaking)
            alpha = 1 - np.exp(-dt / self.attack_time * np.log(2))
        else:
            # Release: value is decreasing (user starting to speak)
            alpha = 1 - np.exp(-dt / self.release_time * np.log(2))
        
        self.value = float((1 - alpha) * self.value + alpha * new_value)
        return self.value
    
    def time_to_decay_to(self, target_value: float) -> float:
        """Calculate time needed to decay from 1.0 to target value.
        
        Args:
            target_value: Value between 0 and 1 to decay to.
            
        Returns:
            Time in seconds.
        """
        assert 0 < target_value < 1
        return float(-self.release_time * np.log2(target_value))
    
    def reset(self, value: float = 0.0):
        """Reset EMA to a specific value."""
        self.value = value


class ConversationState(str, Enum):
    """States for turn-taking state machine."""
    WAITING_FOR_USER = "waiting_for_user"  # Bot finished, waiting for user to speak
    USER_SPEAKING = "user_speaking"         # User is currently speaking
    BOT_SPEAKING = "bot_speaking"           # Bot is currently responding
    PROCESSING = "processing"               # Processing user input, generating response


@dataclass 
class VADConfig:
    """Configuration for Voice Activity Detection."""
    
    # Audio parameters
    sample_rate: int = 16000
    frame_duration_ms: int = 80  # Frame size in milliseconds
    
    # Pause detection thresholds
    pause_threshold: float = 0.6  # Pause prediction above this = user stopped speaking
    interruption_threshold: float = 0.4  # Below this during bot speech = user interrupting
    
    # EMA smoothing parameters (in seconds)
    attack_time: float = 0.01  # Quick response to user stopping
    release_time: float = 0.01  # Quick response to user starting
    
    # Timing parameters
    min_speech_duration_sec: float = 0.3  # Minimum speech before considering pause
    uninterruptible_time_sec: float = 3.0  # Bot can't be interrupted for this long after starting
    silence_timeout_sec: float = 10.0  # Prompt user after this much silence
    
    # Audio energy thresholds (for basic VAD fallback)
    energy_threshold: float = 0.02  # RMS energy above this = speech activity
    
    @property
    def samples_per_frame(self) -> int:
        """Number of samples per audio frame."""
        return int(self.sample_rate * self.frame_duration_ms / 1000)
    
    @property
    def frame_time_sec(self) -> float:
        """Duration of one frame in seconds."""
        return self.frame_duration_ms / 1000


@dataclass
class VADState:
    """Current state of the VAD system."""
    
    conversation_state: ConversationState = ConversationState.WAITING_FOR_USER
    pause_prediction: ExponentialMovingAverage = field(default_factory=lambda: ExponentialMovingAverage(
        attack_time=0.01, release_time=0.01, initial_value=1.0
    ))
    
    # Timing tracking
    last_speech_time: float = 0.0  # When we last detected speech
    speech_start_time: float = 0.0  # When current speech segment started
    bot_start_time: float = 0.0  # When bot started speaking
    waiting_start_time: float = 0.0  # When we started waiting for user
    
    # Counters
    frames_received: int = 0
    total_speech_frames: int = 0
    
    # Flags
    has_received_speech: bool = False  # Has user said anything in current turn
    is_flushing: bool = False  # Waiting for STT to finish processing
    
    def reset_for_new_turn(self):
        """Reset state for a new user turn."""
        self.has_received_speech = False
        self.speech_start_time = 0.0
        self.is_flushing = False
        self.pause_prediction.reset(1.0)  # Start assuming no speech


class VoiceActivityDetector:
    """Semantic Voice Activity Detection for turn-taking.
    
    Uses a combination of:
    1. Energy-based VAD (RMS amplitude)
    2. Pause prediction from STT server (semantic understanding)
    3. State machine for conversation flow
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self.state = VADState()
        self._lock = asyncio.Lock()
        
        # Callbacks
        self._on_speech_started: Optional[Callable[[], None]] = None
        self._on_speech_ended: Optional[Callable[[float], None]] = None  # duration
        self._on_bot_interrupted: Optional[Callable[[], None]] = None
        self._on_silence_timeout: Optional[Callable[[], None]] = None
    
    def set_callbacks(
        self,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_speech_ended: Optional[Callable[[float], None]] = None,
        on_bot_interrupted: Optional[Callable[[], None]] = None,
        on_silence_timeout: Optional[Callable[[], None]] = None,
    ):
        """Set event callbacks for VAD state changes."""
        self._on_speech_started = on_speech_started
        self._on_speech_ended = on_speech_ended
        self._on_bot_interrupted = on_bot_interrupted
        self._on_silence_timeout = on_silence_timeout
    
    async def process_audio_frame(
        self,
        audio: np.ndarray,
        stt_pause_prob: Optional[float] = None,
    ) -> dict:
        """Process an audio frame and update VAD state.
        
        Args:
            audio: Audio samples (float32 normalized or int16)
            stt_pause_prob: Optional pause probability from STT server [0-1]
                           Higher = more likely user has stopped speaking
        
        Returns:
            dict with VAD status and any events triggered
        """
        async with self._lock:
            current_time = time.time()
            events = []
            
            self.state.frames_received += 1
            
            # Convert to float32 if needed
            if audio.dtype == np.int16:
                audio_float = audio.astype(np.float32) / 32767.0
            else:
                audio_float = audio.astype(np.float32)
            
            # Calculate energy-based VAD
            rms_energy = float(np.sqrt(np.mean(audio_float ** 2)))
            has_energy = rms_energy > self.config.energy_threshold
            
            # Update pause prediction if we have STT probability
            if stt_pause_prob is not None:
                self.state.pause_prediction.update(
                    dt=self.config.frame_time_sec,
                    new_value=stt_pause_prob
                )
            elif has_energy:
                # Fallback: use energy to update pause prediction
                # Low pause prob when we detect energy
                self.state.pause_prediction.update(
                    dt=self.config.frame_time_sec,
                    new_value=0.0 if has_energy else 1.0
                )
            
            pause_value = self.state.pause_prediction.value
            
            # State machine transitions
            if self.state.conversation_state == ConversationState.WAITING_FOR_USER:
                # Check for speech start
                if has_energy or (stt_pause_prob is not None and stt_pause_prob < 0.5):
                    self.state.conversation_state = ConversationState.USER_SPEAKING
                    self.state.speech_start_time = current_time
                    self.state.has_received_speech = True
                    self.state.pause_prediction.reset(0.0)
                    events.append("speech_started")
                    
                    if self._on_speech_started:
                        self._on_speech_started()
                    
                    logger.debug("User started speaking")
                
                # Check for silence timeout
                elif (current_time - self.state.waiting_start_time) > self.config.silence_timeout_sec:
                    events.append("silence_timeout")
                    if self._on_silence_timeout:
                        self._on_silence_timeout()
            
            elif self.state.conversation_state == ConversationState.USER_SPEAKING:
                if has_energy:
                    self.state.last_speech_time = current_time
                    self.state.total_speech_frames += 1
                
                # Check for pause (user finished speaking)
                speech_duration = current_time - self.state.speech_start_time
                if (
                    speech_duration > self.config.min_speech_duration_sec
                    and pause_value > self.config.pause_threshold
                    and not self.state.is_flushing
                ):
                    events.append("speech_ended")
                    logger.info(f"User pause detected (prob={pause_value:.2f}, duration={speech_duration:.2f}s)")
                    
                    if self._on_speech_ended:
                        self._on_speech_ended(speech_duration)
            
            elif self.state.conversation_state == ConversationState.BOT_SPEAKING:
                # Check for interruption
                time_since_bot_start = current_time - self.state.bot_start_time
                
                if (
                    time_since_bot_start > self.config.uninterruptible_time_sec
                    and pause_value < self.config.interruption_threshold
                ):
                    events.append("bot_interrupted")
                    logger.info("User interrupted bot")
                    
                    if self._on_bot_interrupted:
                        self._on_bot_interrupted()
            
            return {
                "conversation_state": self.state.conversation_state.value,
                "pause_prediction": pause_value,
                "rms_energy": rms_energy,
                "has_speech_activity": has_energy,
                "events": events,
                "speech_duration": (current_time - self.state.speech_start_time) 
                    if self.state.speech_start_time > 0 else 0,
            }
    
    def transition_to_bot_speaking(self):
        """Transition to bot speaking state."""
        self.state.conversation_state = ConversationState.BOT_SPEAKING
        self.state.bot_start_time = time.time()
        logger.debug("Transitioned to BOT_SPEAKING")
    
    def transition_to_waiting(self):
        """Transition to waiting for user state."""
        self.state.conversation_state = ConversationState.WAITING_FOR_USER
        self.state.waiting_start_time = time.time()
        self.state.reset_for_new_turn()
        logger.debug("Transitioned to WAITING_FOR_USER")
    
    def transition_to_processing(self):
        """Transition to processing state (generating response)."""
        self.state.conversation_state = ConversationState.PROCESSING
        self.state.is_flushing = True
        logger.debug("Transitioned to PROCESSING")
    
    def get_state(self) -> dict:
        """Get current VAD state as dictionary."""
        return {
            "conversation_state": self.state.conversation_state.value,
            "pause_prediction": self.state.pause_prediction.value,
            "has_received_speech": self.state.has_received_speech,
            "frames_received": self.state.frames_received,
            "is_flushing": self.state.is_flushing,
        }
    
    def should_respond(self) -> bool:
        """Check if the bot should start responding now."""
        return (
            self.state.conversation_state == ConversationState.USER_SPEAKING
            and self.state.has_received_speech
            and self.state.pause_prediction.value > self.config.pause_threshold
            and not self.state.is_flushing
        )


class TurnManager:
    """High-level turn-taking manager for conversation flow.
    
    Coordinates between VAD, STT, and response generation.
    """
    
    def __init__(
        self,
        vad: VoiceActivityDetector,
        on_turn_complete: Callable[[str], None],
    ):
        self.vad = vad
        self._on_turn_complete = on_turn_complete
        self._transcript_buffer: list[str] = []
        self._current_turn_id: int = 0
    
    def add_transcript(self, text: str, is_final: bool = False):
        """Add transcript text from STT."""
        if text.strip():
            self._transcript_buffer.append(text.strip())
        
        if is_final and self._transcript_buffer:
            # Check if we should end the turn
            if self.vad.should_respond():
                self._end_turn()
    
    def _end_turn(self):
        """End the current user turn and trigger response generation."""
        if not self._transcript_buffer:
            return
        
        full_text = " ".join(self._transcript_buffer)
        self._transcript_buffer = []
        self._current_turn_id += 1
        
        self.vad.transition_to_processing()
        self._on_turn_complete(full_text)
    
    def force_end_turn(self):
        """Force end the current turn (e.g., on explicit stop)."""
        self._end_turn()
    
    def start_new_turn(self):
        """Prepare for a new user turn."""
        self._transcript_buffer = []
        self.vad.transition_to_waiting()
    
    def on_response_complete(self):
        """Called when bot finishes responding."""
        self.start_new_turn()


# Utility functions
def calculate_audio_rms(audio: np.ndarray) -> float:
    """Calculate RMS energy of audio samples."""
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32767.0
    return float(np.sqrt(np.mean(audio ** 2)))


def detect_silence(audio: np.ndarray, threshold: float = 0.02) -> bool:
    """Simple silence detection based on RMS energy."""
    return calculate_audio_rms(audio) < threshold


# Test
async def test_vad():
    """Simple VAD test."""
    logging.basicConfig(level=logging.DEBUG)
    
    vad = VoiceActivityDetector()
    
    # Simulate some audio frames
    for i in range(20):
        # Alternate between silence and speech
        if 5 <= i <= 15:
            audio = np.random.randn(1280).astype(np.float32) * 0.1
        else:
            audio = np.zeros(1280, dtype=np.float32)
        
        result = await vad.process_audio_frame(audio)
        print(f"Frame {i}: state={result['conversation_state']}, "
              f"pause={result['pause_prediction']:.2f}, "
              f"events={result['events']}")


if __name__ == "__main__":
    asyncio.run(test_vad())
