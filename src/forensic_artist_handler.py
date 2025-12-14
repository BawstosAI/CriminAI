"""Real-time Forensic Artist Orchestration Handler.

Orchestrates the full pipeline: STT -> Gemini LLM -> TTS -> Image Generation
Based on kyutai-labs/unmute architecture.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("Install openai: pip install openai")

from stt_service import SpeechToText, STTConfig, TranscriptionResult, MicrophoneSTT
from tts_service import TextToSpeech, TTSConfig

logger = logging.getLogger(__name__)


@dataclass
class ForensicArtistConfig:
    """Configuration for the Forensic Artist system."""
    # Service endpoints
    stt_host: str = "localhost"
    stt_port: int = 8090
    llm_host: str = "localhost"
    llm_port: int = 8091
    
    # TTS configuration (now using Gradium)
    tts_api_key: str = ""  # Gradium API key (loaded from GRADIUM_API_KEY env var)
    tts_region: str = "eu"  # "eu" or "us"
    tts_voice_id: str = "YTpq7expH9539ERJ"  # Emma (default English voice)
    
    # Model settings
    llm_model: str = "gemini-2.0-flash"
    temperature: float = 0.9
    max_tokens: int = 1024
    
    # Audio settings
    sample_rate: int = 16000
    silence_threshold: float = 0.01
    silence_duration: float = 1.5
    
    # System prompt for forensic artist
    system_prompt: str = """You are an expert police forensic sketch artist interviewing a witness.
Goals:
  - Ask one concise question at a time covering facial shape, skin tone, eyes,
    brows, nose, mouth, hair, facial hair, accessories, and unique marks.
  - Maintain a JSON object named suspect_profile that captures every detail.
  - Keep responses professional and focused on fact gathering.
  - When the witness finishes, confirm the summary and craft a final,
    diffusion-ready charcoal sketch prompt referencing every gathered fact in
    no more than ~70 tokens. Merge adjectives where possible.

Format every reply exactly as:
RESPONSE: <spoken reply and next question>
STATE: {"suspect_profile": {...}}
PROMPT: <"PENDING" while collecting info, final prompt when ready>
"""

    # Interview state tracking fields
    questions_asked: int = 0
    max_questions: int = 15


@dataclass
class ConversationMessage:
    """A message in the conversation."""
    role: str  # "user" or "assistant"
    content: str


@dataclass 
class ForensicArtistState:
    """State of the forensic artist interview."""
    messages: list[ConversationMessage] = field(default_factory=list)
    is_complete: bool = False
    sketch_prompt: Optional[str] = None
    suspect_profile: Optional[dict] = None
    questions_asked: int = 0


class ForensicArtistHandler:
    """Main handler orchestrating STT -> LLM -> TTS -> Image Generation."""

    def __init__(self, config: Optional[ForensicArtistConfig] = None):
        self.config = config or ForensicArtistConfig()
        
        # Initialize services
        self.stt = SpeechToText(STTConfig(
            server_host=self.config.stt_host,
            server_port=self.config.stt_port,
        ))
        
        self.tts = TextToSpeech(TTSConfig(
            api_key=self.config.tts_api_key,
            region=self.config.tts_region,
            voice_id=self.config.tts_voice_id,
        ))
        
        # OpenAI-compatible client pointing to Gemini wrapper
        self.llm = AsyncOpenAI(
            base_url=f"http://{self.config.llm_host}:{self.config.llm_port}/v1",
            api_key="not-needed",  # Gemini wrapper handles auth
        )
        
        self.state = ForensicArtistState()
        self._running = False
        
        # Callbacks
        self.on_user_speech: Optional[Callable[[str], None]] = None
        self.on_assistant_speech: Optional[Callable[[str], None]] = None
        self.on_sketch_ready: Optional[Callable[[str], None]] = None

    async def start(self) -> None:
        """Start the forensic artist session."""
        self._running = True
        self.state = ForensicArtistState()
        logger.info("Forensic Artist session started")

    async def stop(self) -> None:
        """Stop the session."""
        self._running = False
        await self.stt.disconnect()
        await self.tts.disconnect()
        logger.info("Forensic Artist session stopped")

    async def _get_llm_response(self, user_message: str) -> AsyncIterator[str]:
        """Get streaming response from LLM."""
        # Add user message to history
        self.state.messages.append(ConversationMessage(role="user", content=user_message))
        
        # Build messages for API
        messages = [{"role": "system", "content": self.config.system_prompt}]
        for msg in self.state.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Stream response
        response = await self.llm.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        
        full_response = []
        async for chunk in response:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_response.append(text)
                yield text
        
        # Store assistant response
        assistant_message = "".join(full_response)
        self.state.messages.append(ConversationMessage(role="assistant", content=assistant_message))
        self.state.questions_asked += 1
        
        # Parse structured response: RESPONSE:, STATE:, PROMPT:
        parsed = self._parse_response(assistant_message)
        if parsed.get("prompt") and parsed["prompt"] != "PENDING":
            self.state.is_complete = True
            self.state.sketch_prompt = parsed["prompt"]
            if self.on_sketch_ready:
                self.on_sketch_ready(self.state.sketch_prompt)
        
        # Store state if present
        if parsed.get("state"):
            self.state.suspect_profile = parsed["state"]

    def _parse_response(self, text: str) -> dict:
        """Parse structured response with RESPONSE:, STATE:, PROMPT: sections."""
        import re
        import json
        
        result = {"response": "", "state": None, "prompt": None}
        
        # Extract RESPONSE section
        response_match = re.search(r'RESPONSE:\s*(.+?)(?=STATE:|PROMPT:|$)', text, re.DOTALL)
        if response_match:
            result["response"] = response_match.group(1).strip()
        
        # Extract STATE section (JSON)
        state_match = re.search(r'STATE:\s*(\{.+?\})(?=PROMPT:|$)', text, re.DOTALL)
        if state_match:
            try:
                result["state"] = json.loads(state_match.group(1))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse STATE JSON: {state_match.group(1)}")
        
        # Extract PROMPT section
        prompt_match = re.search(r'PROMPT:\s*(.+?)$', text, re.DOTALL)
        if prompt_match:
            prompt_text = prompt_match.group(1).strip()
            if prompt_text.upper() != "PENDING":
                result["prompt"] = prompt_text
        
        return result

    async def process_user_speech(self, transcript: str) -> AsyncIterator[str]:
        """Process user speech and generate assistant response.
        
        Args:
            transcript: User's speech transcription
            
        Yields:
            Assistant response text chunks for TTS
        """
        if self.on_user_speech:
            self.on_user_speech(transcript)
        
        logger.info(f"User: {transcript}")
        
        async for chunk in self._get_llm_response(transcript):
            if self.on_assistant_speech:
                self.on_assistant_speech(chunk)
            yield chunk

    async def run_turn(self, user_input: str) -> str:
        """Run a single conversation turn (text mode).
        
        Args:
            user_input: User's text input
            
        Returns:
            Assistant's complete response
        """
        chunks = []
        async for chunk in self.process_user_speech(user_input):
            chunks.append(chunk)
        return "".join(chunks)

    async def run_voice_turn(self) -> tuple[str, str]:
        """Run a single voice conversation turn.
        
        Returns:
            Tuple of (user_transcript, assistant_response)
        """
        # Listen for user speech
        mic = MicrophoneSTT(self.stt)
        
        transcripts = []
        def on_transcript(result: TranscriptionResult):
            if result.is_final:
                transcripts.append(result.text)
        
        user_transcript = await mic.start(
            on_transcript,
            silence_threshold=self.config.silence_threshold,
            silence_duration=self.config.silence_duration,
        )
        
        if not user_transcript.strip():
            return "", ""
        
        # Get LLM response and synthesize to speech
        response_chunks = []
        
        async def response_stream():
            async for chunk in self.process_user_speech(user_transcript):
                response_chunks.append(chunk)
                yield chunk
        
        # Synthesize and play response
        await self.tts.synthesize_and_play(response_stream())
        
        return user_transcript, "".join(response_chunks)

    async def run_voice_session(self) -> ForensicArtistState:
        """Run full voice interview session.
        
        Returns:
            Final state with conversation history and sketch prompt
        """
        await self.start()
        
        # Initial greeting
        greeting = "Hello! I'm your AI forensic artist. I'm here to help create a composite sketch based on your description. Let's start with the basics - can you tell me about the general build and approximate age of the person you saw?"
        
        if self.on_assistant_speech:
            self.on_assistant_speech(greeting)
        
        # Synthesize greeting
        async def greeting_stream():
            yield greeting
        
        await self.tts.synthesize_and_play(greeting_stream())
        self.state.messages.append(ConversationMessage(role="assistant", content=greeting))
        
        # Main interview loop
        while self._running and not self.state.is_complete:
            if self.state.questions_asked >= self.config.max_questions:
                logger.info("Max questions reached, ending interview")
                break
            
            try:
                user_text, assistant_text = await self.run_voice_turn()
                
                if not user_text:
                    continue
                
                logger.info(f"Turn {self.state.questions_asked}: User={user_text[:50]}... Assistant={assistant_text[:50]}...")
                
            except Exception as e:
                logger.error(f"Error in voice turn: {e}")
                break
        
        await self.stop()
        return self.state


class TextForensicArtist:
    """Text-only version of the forensic artist (no audio)."""

    def __init__(self, config: Optional[ForensicArtistConfig] = None):
        self.config = config or ForensicArtistConfig()
        
        # OpenAI-compatible client
        self.llm = AsyncOpenAI(
            base_url=f"http://{self.config.llm_host}:{self.config.llm_port}/v1",
            api_key="not-needed",
        )
        
        self.state = ForensicArtistState()

    async def chat(self, user_message: str) -> str:
        """Send a message and get response."""
        self.state.messages.append(ConversationMessage(role="user", content=user_message))
        
        messages = [{"role": "system", "content": self.config.system_prompt}]
        for msg in self.state.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        response = await self.llm.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.temperature,
        )
        
        assistant_message = response.choices[0].message.content
        self.state.messages.append(ConversationMessage(role="assistant", content=assistant_message))
        self.state.questions_asked += 1
        
        # Parse structured response
        parsed = self._parse_response(assistant_message)
        if parsed.get("prompt") and parsed["prompt"] != "PENDING":
            self.state.is_complete = True
            self.state.sketch_prompt = parsed["prompt"]
        if parsed.get("state"):
            self.state.suspect_profile = parsed["state"]
        
        # Return just the RESPONSE part for display
        return parsed.get("response") or assistant_message

    def _parse_response(self, text: str) -> dict:
        """Parse structured response with RESPONSE:, STATE:, PROMPT: sections."""
        import re
        import json
        
        result = {"response": "", "state": None, "prompt": None}
        
        # Extract RESPONSE section
        response_match = re.search(r'RESPONSE:\s*(.+?)(?=STATE:|PROMPT:|$)', text, re.DOTALL)
        if response_match:
            result["response"] = response_match.group(1).strip()
        
        # Extract STATE section (JSON)
        state_match = re.search(r'STATE:\s*(\{.+?\})(?=PROMPT:|$)', text, re.DOTALL)
        if state_match:
            try:
                result["state"] = json.loads(state_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Extract PROMPT section
        prompt_match = re.search(r'PROMPT:\s*(.+?)$', text, re.DOTALL)
        if prompt_match:
            prompt_text = prompt_match.group(1).strip()
            if prompt_text.upper() != "PENDING":
                result["prompt"] = prompt_text
        
        return result

    async def run_session(self) -> ForensicArtistState:
        """Run interactive text session."""
        print("\n" + "="*60)
        print("AI FORENSIC SKETCH ARTIST")
        print("="*60)
        print("\nHello! I'm your AI forensic artist.")
        print("Describe the person you saw, and I'll create a composite sketch.")
        print("Type 'quit' to exit.\n")
        
        greeting = "Let's start with the basics - can you tell me about the general build and approximate age of the person you saw?"
        print(f"Artist: {greeting}\n")
        self.state.messages.append(ConversationMessage(role="assistant", content=greeting))
        
        while not self.state.is_complete:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            response = await self.chat(user_input)
            print(f"\nArtist: {response}\n")
            
            if self.state.is_complete:
                print("\n" + "="*60)
                print("SKETCH PROMPT READY:")
                print(self.state.sketch_prompt)
                print("="*60 + "\n")
        
        return self.state


# CLI entry point
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Forensic Artist Handler")
    parser.add_argument("--mode", choices=["text", "voice"], default="text", help="Interview mode")
    parser.add_argument("--llm-host", default="localhost", help="LLM server host")
    parser.add_argument("--llm-port", type=int, default=8091, help="LLM server port")
    parser.add_argument("--stt-port", type=int, default=8090, help="STT server port")
    parser.add_argument("--tts-port", type=int, default=8089, help="TTS server port")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    config = ForensicArtistConfig(
        llm_host=args.llm_host,
        llm_port=args.llm_port,
        stt_port=args.stt_port,
        tts_port=args.tts_port,
    )
    
    if args.mode == "voice":
        handler = ForensicArtistHandler(config)
        state = await handler.run_voice_session()
    else:
        artist = TextForensicArtist(config)
        state = await artist.run_session()
    
    if state.sketch_prompt:
        print(f"\nFinal sketch prompt: {state.sketch_prompt}")
        
        # Optionally generate image
        generate = input("\nGenerate sketch image? [y/N]: ").strip().lower()
        if generate == 'y':
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from image_gen import generate_image
                
                output_path = generate_image(state.sketch_prompt)
                print(f"Sketch saved to: {output_path}")
            except Exception as e:
                print(f"Image generation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
