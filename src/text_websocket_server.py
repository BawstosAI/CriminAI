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


# System prompt for forensic artist
SYSTEM_PROMPT = """You are an AI forensic sketch artist assistant. Your name is CrimnAI and your job start as soon as we call you like "Hey CrimnAI". Your job is to interview a witness to gather details about a suspect's appearance and generate a detailed text prompt for image generation.

INTERVIEW GUIDELINES:
1. Ask ONE question at a time
2. Focus on specific facial features: face shape, eyes, nose, mouth, hair, distinguishing marks
3. Be conversational and supportive - witnesses may be stressed
4. After gathering sufficient details (about 8-10 questions), summarize and confirm

WHEN YOU HAVE ENOUGH DETAILS:
End your final message with a special marker containing the image prompt:
[SKETCH_PROMPT: detailed description here]

The prompt should be a single paragraph describing all gathered facial features suitable for image generation.

START by greeting the witness and asking about the overall face shape."""


@dataclass
class Session:
    """Tracks a single user session."""
    session_id: str
    messages: list = field(default_factory=list)
    questions_asked: int = 0
    is_complete: bool = False
    sketch_prompt: Optional[str] = None


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
            # Send initial greeting
            greeting = await self._get_initial_greeting(session)
            await self._send_message(websocket, "assistant_message", greeting)
            
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
    
    async def _get_initial_greeting(self, session: Session) -> str:
        """Get the initial greeting from the LLM."""
        session.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        
        try:
            response = await self.client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=session.messages,
                max_tokens=300,
            )
            greeting = response.choices[0].message.content
            session.messages.append({"role": "assistant", "content": greeting})
            return greeting
        except Exception as e:
            logger.error(f"LLM error: {e}")
            fallback = (
                "Hello! I'm your AI forensic artist. Let's create a composite sketch. "
                "Can you describe the overall shape of the suspect's face? "
                "Was it round, oval, square, or something else?"
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
            
            # Add user message
            session.messages.append({"role": "user", "content": user_text})
            session.questions_asked += 1
            
            # Send typing indicator
            await self._send_message(websocket, "typing", None)
            
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
                await self._send_error(websocket, f"LLM error: {e}")
        
        elif msg_type == "generate_image":
            # User wants to generate the image
            prompt = data.get("prompt") or session.sketch_prompt
            if not prompt:
                await self._send_error(websocket, "No sketch prompt available")
                return
            
            await self._send_message(websocket, "generating_image", "Starting image generation...")
            
            try:
                import base64
                from image_gen import generate_image_api, get_demo_image_path
                
                demo_image = get_demo_image_path()
                image_path = demo_image or generate_image_api(prompt)
                
                # Read image and convert to base64 for browser display
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                
                # Send as base64 data URL
                image_url = f"data:image/png;base64,{image_data}"
                
                logger.info(f"Image generated: {image_path}")
                
                # Send success with base64 URL
                await self._send_message(
                    websocket,
                    "image_ready",
                    prompt,
                    extra={"image_url": image_url, "image_path": image_path}
                )
            except Exception as e:
                logger.exception(f"Image generation error: {e}")
                await self._send_error(websocket, f"Image generation failed: {e}")
        
        elif msg_type == "ping":
            await self._send_message(websocket, "pong", None)
    
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
