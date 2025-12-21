"""WebSocket server for text-based forensic artist interaction.

Provides real-time text chat with the forensic artist backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env")

try:
    import websockets
    from websockets.asyncio.server import serve, ServerConnection
except ImportError:
    raise ImportError("Install websockets: pip install websockets")

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("Install openai: pip install openai")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# System prompt (aligned with audio mode)
SYSTEM_PROMPT = """You are an AI forensic sketch artist assistant. Your name is CrimnAI and your job starts as soon as we hear or read "Hey CrimnAI" (tolerate minor variations). Your job is to interview a witness to gather details about a suspect's appearance and generate a detailed text prompt for image generation.

INTERVIEW GUIDELINES:
1. Ask ONE question at a time
2. Focus on specific facial features: face shape, eyes, nose, mouth, hair, distinguishing marks
3. Be conversational and supportive - witnesses may be stressed
4. After gathering sufficient details (about 8-10 questions), summarize and confirm

WHEN YOU HAVE ENOUGH DETAILS:
End your final message with a special marker containing the image prompt:
[SKETCH_PROMPT: detailed description here]

The prompt should be a single paragraph describing all gathered facial features suitable for image generation.

START behavior:
- Wait until the user types something; once we see "Hey CrimnAI" (or variations), greet and begin the interview.
- Automatically generate the image when the [SKETCH_PROMPT] is produced; then ask if the sketch is correct.
- If the user says it's correct, thank them for their collaboration and promise that the police will catch the criminal.
- If the user says it's not correct, ask clarifying questions about what to change, produce a new [SKETCH_PROMPT], and repeat until the user is satisfied."""


@dataclass
class Session:
    """Tracks a single user session."""
    session_id: str
    messages: list = field(default_factory=list)
    questions_asked: int = 0
    is_complete: bool = False
    sketch_prompt: Optional[str] = None
    awaiting_confirmation: bool = False
    is_awake: bool = False
    image_in_progress: bool = False
    language: str = "English"
    wake_attempts: int = 0
    wake_attempts: int = 0
    image_in_progress: bool = False


class TextArtistServer:
    """WebSocket server for text-based forensic artist."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8092,
        llm_host: str = "localhost",
        llm_port: int = 8091,
    ):
        self.host = host
        self.port = port
        self.llm_host = llm_host
        self.llm_port = llm_port
        self.sessions: dict[str, Session] = {}
        
        self.client = AsyncOpenAI(
            base_url=f"http://{llm_host}:{llm_port}/v1",
            api_key="not-needed",
        )
        
    async def handle_connection(self, websocket: ServerConnection):
        """Handle a WebSocket connection."""
        session_id = str(id(websocket))
        session = Session(session_id=session_id)
        self.sessions[session_id] = session
        
        logger.info(f"New connection: {session_id}")
        
        try:
            # Initial instruction; wait for wake message
            await self._send_message(
                websocket,
                "assistant_message",
                "CrimnAI is listening. Type 'Hey CrimnAI' to begin."
            )
            
            # Handle incoming messages
            async for raw_message in websocket:
                try:
                    data = json.loads(raw_message)
                    await self._handle_message(websocket, session, data)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON")
                except Exception as e:
                    logger.exception(f"Error handling message: {e}")
                    await self._send_error(websocket, str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {session_id}")
        finally:
            del self.sessions[session_id]
    
    async def _get_greeting_after_wake(self, session: Session) -> str:
        """Get greeting after wake message is detected."""
        if not session.messages or session.messages[0].get("role") != "system":
            lang = session.language or "English"
            system_prompt = SYSTEM_PROMPT + f"\nAlways respond in {lang}."
            session.messages.insert(0, {"role": "system", "content": system_prompt})
        has_user = any(m.get("role") == "user" for m in session.messages)
        if not has_user:
            session.messages.append({"role": "user", "content": "Hey CrimnAI"})
        try:
            response = await self.client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=session.messages,
                max_tokens=200,
            )
            greeting = response.choices[0].message.content
            session.messages.append({"role": "assistant", "content": greeting})
            return greeting
        except Exception as e:
            logger.error(f"LLM error: {e}")
            fallback = (
                "Hello! I'm CrimnAI. Let's build a composite sketch. "
                "Tell me the overall face shape to begin."
            )
            session.messages.append({"role": "assistant", "content": fallback})
            return fallback
    
    async def _handle_message(
        self, 
        websocket: ServerConnection, 
        session: Session, 
        data: dict
    ):
        """Process an incoming message."""
        msg_type = data.get("type")
        
        if msg_type == "user_message":
            user_text = data.get("text", "").strip()
            if not user_text:
                return

            # Wake-word handling
            if not session.is_awake:
                lowered = user_text.lower()
                wake_words = ["crimnai", "crim ai", "criminai", "crim n ai", "hey crim", "hey krem"]
                session.wake_attempts += 1
                if any(w in lowered for w in wake_words) or session.wake_attempts >= 3:
                    session.is_awake = True
                    session.wake_attempts = 0
                    session.messages.append({"role": "user", "content": user_text})
                    greeting = await self._get_greeting_after_wake(session)
                    await self._send_message(websocket, "assistant_message", greeting)
                else:
                    await self._send_message(websocket, "assistant_message", "Say 'Hey CrimnAI' to begin.")
                return

            # Add user message
            session.messages.append({"role": "user", "content": user_text})
            session.questions_asked += 1
            
            # Send typing indicator
            await self._send_message(websocket, "typing", None)

            # Confirmation loop
            if session.awaiting_confirmation:
                try:
                    response = await self.client.chat.completions.create(
                        model="gemini-2.0-flash",
                        messages=session.messages,
                        max_tokens=400,
                    )
                    assistant_text = response.choices[0].message.content
                    session.messages.append({"role": "assistant", "content": assistant_text})
                    if "[SKETCH_PROMPT:" in assistant_text:
                        start = assistant_text.find("[SKETCH_PROMPT:") + len("[SKETCH_PROMPT:")
                        end = assistant_text.find("]", start)
                        if end > start:
                            session.sketch_prompt = assistant_text[start:end].strip()
                            session.is_complete = True
                            clean_text = assistant_text[:assistant_text.find("[SKETCH_PROMPT:")].strip()
                            await self._send_message(websocket, "assistant_message", clean_text)
                            await self._start_image_generation(websocket, session, session.sketch_prompt)
                            return
                    if "yes" in user_text.lower() and "no" not in user_text.lower():
                        thanks = (
                            "Thank you for your collaboration and bravery. We will catch this criminal."
                        )
                        session.awaiting_confirmation = False
                        await self._send_message(websocket, "assistant_message", thanks)
                        return
                    await self._send_message(websocket, "assistant_message", assistant_text)
                except Exception as e:
                    logger.exception(f"LLM error: {e}")
                    fallback = (
                        "I didn't catch that. Tell me what to fix in the sketch (face shape, eyes, nose, mouth, hair, marks)."
                    )
                    await self._send_message(websocket, "assistant_message", fallback)
                return
            
            # Get LLM response
            try:
                response = await self.client.chat.completions.create(
                    model="gemini-2.0-flash",
                    messages=session.messages,
                    max_tokens=500,
                )
                assistant_text = response.choices[0].message.content
                session.messages.append({"role": "assistant", "content": assistant_text})
                
                # Check for sketch prompt
                if "[SKETCH_PROMPT:" in assistant_text:
                    start = assistant_text.find("[SKETCH_PROMPT:") + len("[SKETCH_PROMPT:")
                    end = assistant_text.find("]", start)
                    if end > start:
                        session.sketch_prompt = assistant_text[start:end].strip()
                        session.is_complete = True
                        session.awaiting_confirmation = True
                        
                        # Send response without the marker
                        clean_text = assistant_text[:assistant_text.find("[SKETCH_PROMPT:")].strip()
                        await self._send_message(websocket, "assistant_message", clean_text)
                        
                        # Send completion with prompt
                        await self._send_message(
                            websocket, 
                            "interview_complete", 
                            session.sketch_prompt,
                            extra={"questions_asked": session.questions_asked}
                        )
                        # Auto-start image generation server-side
                        await self._start_image_generation(websocket, session, session.sketch_prompt)
                        return
                
                # Send regular response
                await self._send_message(
                    websocket, 
                    "assistant_message", 
                    assistant_text,
                    extra={"questions_asked": session.questions_asked}
                )
                
            except Exception as e:
                logger.exception(f"LLM error: {e}")
                # Graceful fallback so UI isn't stuck
                fallback = (
                    "I'm having trouble reaching the language model right now. "
                    "Please continue describing the suspect's face shape, eyes, nose, mouth, hair, and any marks."
                )
                session.messages.append({"role": "assistant", "content": fallback})
                await self._send_message(websocket, "assistant_message", fallback)
        
        elif msg_type == "generate_image":
            # User wants to generate the image
            prompt = data.get("prompt") or session.sketch_prompt
            if not prompt:
                await self._send_error(websocket, "No sketch prompt available")
                return
            await self._start_image_generation(websocket, session, prompt)
        
        elif msg_type == "ping":
            await self._send_message(websocket, "pong", None)

        elif msg_type == "language_selected":
            lang = data.get("language", "English")
            session.language = lang
            # Do not auto-awaken; still wait for wake word
            system_prompt = SYSTEM_PROMPT + f"\nAlways respond in {lang}."
            session.messages = [{"role": "system", "content": system_prompt}]
    
    async def _send_message(
        self, 
        websocket: ServerConnection, 
        msg_type: str, 
        text: Optional[str],
        extra: Optional[dict] = None
    ):
        """Send a message to the client."""
        payload = {"type": msg_type}
        if text is not None:
            payload["text"] = text
        if extra:
            payload.update(extra)
        await websocket.send(json.dumps(payload))
    
    async def _send_error(self, websocket: ServerConnection, error: str):
        """Send an error message."""
        await self._send_message(websocket, "error", error)

    async def _start_image_generation(self, websocket: ServerConnection, session: Session, prompt: str):
        """Generate image (demo-first) and push to client; avoid duplicate runs."""
        if session.image_in_progress:
            return
        session.image_in_progress = True
        await self._send_message(websocket, "generating_image", "Starting image generation...")
        try:
            import base64
            from image_gen import generate_image_api, get_demo_image_path

            demo_image = get_demo_image_path()
            image_path = demo_image or generate_image_api(prompt)

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            image_url = f"data:image/png;base64,{image_data}"

            logger.info(f"Image generated: {image_path}")

            await self._send_message(
                websocket,
                "image_ready",
                prompt,
                extra={"image_url": image_url, "image_path": image_path}
            )
        except Exception as e:
            logger.exception(f"Image generation error: {e}")
            await self._send_error(websocket, f"Image generation failed: {e}")
        finally:
            session.image_in_progress = False
    
    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"Starting Text Artist WebSocket server on ws://{self.host}:{self.port}")
        
        async with serve(self.handle_connection, self.host, self.port):
            await asyncio.Future()  # Run forever


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Text Artist WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--llm-host", default=os.environ.get("LLM_HOST", "localhost"))
    parser.add_argument("--llm-port", type=int, default=int(os.environ.get("LLM_PORT", "8091")))
    args = parser.parse_args()
    
    server = TextArtistServer(
        host=args.host,
        port=args.port,
        llm_host=args.llm_host,
        llm_port=args.llm_port,
    )
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
