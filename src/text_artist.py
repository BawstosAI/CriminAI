"""Text-only Forensic Artist - no audio dependencies.

Simple text chat with the LLM for forensic sketch interviews.
Supports image generation with FLUX.1-dev.

Modes:
  - dev: Shows prompt, waits for user confirmation before generating
  - prod: Automatically generates image when interview is complete
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("Install openai: pip install openai")

logger = logging.getLogger(__name__)


class Mode(str, Enum):
    """Operating mode for the artist."""
    DEV = "dev"    # Show prompt, wait for confirmation
    PROD = "prod"  # Auto-generate image


@dataclass
class ArtistConfig:
    """Configuration for the text artist."""
    llm_host: str = "localhost"
    llm_port: int = 8091
    llm_model: str = "gemini-2.0-flash"
    temperature: float = 0.9
    max_tokens: int = 1024
    
    # Operating mode
    mode: Mode = Mode.DEV
    
    # Image generation settings
    use_api: bool = True  # Use HF API (True) or local model (False)
    
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

    max_questions: int = 15


@dataclass
class Message:
    """A conversation message."""
    role: str  # "user" or "assistant"
    content: str


@dataclass 
class ArtistState:
    """State of the interview."""
    messages: list[Message] = field(default_factory=list)
    is_complete: bool = False
    sketch_prompt: Optional[str] = None
    suspect_profile: Optional[dict] = None
    questions_asked: int = 0


class TextArtist:
    """Text-only forensic artist."""

    def __init__(self, config: Optional[ArtistConfig] = None):
        self.config = config or ArtistConfig()
        
        # OpenAI-compatible client
        self.llm = AsyncOpenAI(
            base_url=f"http://{self.config.llm_host}:{self.config.llm_port}/v1",
            api_key="not-needed",
        )
        
        self.state = ArtistState()

    async def chat(self, user_message: str) -> str:
        """Send a message and get response."""
        self.state.messages.append(Message(role="user", content=user_message))
        
        # Build messages for API
        messages = [{"role": "system", "content": self.config.system_prompt}]
        for msg in self.state.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Call LLM
        response = await self.llm.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.temperature,
        )
        
        assistant_message = response.choices[0].message.content
        self.state.messages.append(Message(role="assistant", content=assistant_message))
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


async def main():
    """Interactive CLI session."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Forensic Artist CLI")
    parser.add_argument("--llm-host", default=os.environ.get("LLM_HOST", "localhost"))
    parser.add_argument("--llm-port", type=int, default=int(os.environ.get("LLM_PORT", "8091")))
    parser.add_argument(
        "--mode", 
        choices=["dev", "prod"], 
        default="dev",
        help="dev: show prompt & wait for 'send' | prod: auto-generate image"
    )
    parser.add_argument(
        "--local-gpu",
        action="store_true",
        help="Use local GPU for image generation instead of HF API"
    )
    args = parser.parse_args()

    config = ArtistConfig(
        llm_host=args.llm_host, 
        llm_port=args.llm_port,
        mode=Mode(args.mode),
        use_api=not args.local_gpu,
    )
    artist = TextArtist(config)
    
    print("\n" + "=" * 60)
    print("FORENSIC ARTIST - TEXT MODE")
    print("=" * 60)
    print(f"\nLLM: http://{config.llm_host}:{config.llm_port}")
    print(f"Mode: {config.mode.value.upper()}")
    if config.mode == Mode.DEV:
        print("  (Will show prompt and wait for 'send' to generate)")
    else:
        print("  (Will auto-generate image when complete)")
    print("\nType 'quit' to exit.\n")

    # Initial greeting
    greeting = (
        "Hello! I'm your AI forensic artist. Let's create a composite sketch. "
        "Can you describe the overall shape of the suspect's face? "
        "Was it round, oval, square, or something else?"
    )
    print(f"Artist: {greeting}\n")
    artist.state.messages.append(Message(role="assistant", content=greeting))
    
    while not artist.state.is_complete:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            return
            
        if not user_input:
            continue
            
        if user_input.lower() in ['quit', 'exit', 'q']:
            return
        
        try:
            response = await artist.chat(user_input)
            print(f"\nArtist: {response}")
            print(f"  [Q: {artist.state.questions_asked}]\n")
            
        except Exception as e:
            print(f"\nError: {e}")
            print(f"Is the LLM server running at http://{config.llm_host}:{config.llm_port}?\n")
    
    # Interview complete - handle image generation based on mode
    if artist.state.is_complete and artist.state.sketch_prompt:
        print("\n" + "=" * 60)
        print("INTERVIEW COMPLETE!")
        print("=" * 60)
        
        prompt = artist.state.sketch_prompt
        
        if config.mode == Mode.DEV:
            # DEV mode: Show prompt and wait for confirmation
            print(f"\n📝 Generated Prompt:\n")
            print("-" * 40)
            print(prompt)
            print("-" * 40)
            print("\nType 'send' to generate the image, or 'edit' to modify the prompt.")
            
            while True:
                try:
                    cmd = input("\nCommand: ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\nCancelled.")
                    return
                
                if cmd == "send":
                    break
                elif cmd == "edit":
                    print("Enter new prompt (or press Enter to keep current):")
                    new_prompt = input("> ").strip()
                    if new_prompt:
                        prompt = new_prompt
                        print(f"Updated prompt: {prompt}")
                elif cmd in ["quit", "exit", "q"]:
                    print("Cancelled.")
                    return
                else:
                    print("Commands: 'send' to generate, 'edit' to modify, 'quit' to cancel")
        
        else:
            # PROD mode: Auto-generate
            print(f"\nGenerating forensic sketch...")
        
        # Generate the image
        try:
            from image_gen import generate_image, generate_image_api
            
            print("\n⏳ Generating image with FLUX.1-dev...")
            print("   (This may take a minute...)\n")
            
            if config.use_api:
                image_path = generate_image_api(prompt)
            else:
                image_path = generate_image(prompt)
            
            print("=" * 60)
            print("✅ IMAGE GENERATED!")
            print("=" * 60)
            print(f"\n📁 Saved to: {image_path}\n")
            
        except ImportError as e:
            print(f"\n❌ Missing dependencies: {e}")
            print("Install with: pip install diffusers torch huggingface_hub")
        except Exception as e:
            print(f"\n❌ Image generation failed: {e}")
            logger.exception("Image generation error")


if __name__ == "__main__":
    asyncio.run(main())
