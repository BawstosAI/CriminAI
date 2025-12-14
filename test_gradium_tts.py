#!/usr/bin/env python3
"""Test Gradium TTS service integration."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tts_service import TextToSpeech, TTSConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

async def test_tts():
    """Test Gradium TTS connection and synthesis."""
    
    # Check for API key
    api_key = os.environ.get("GRADIUM_API_KEY", "")
    if not api_key:
        print("\n❌ Error: GRADIUM_API_KEY environment variable not set")
        print("\n📝 To get a Gradium API key:")
        print("   1. Visit https://gradium.ai/")
        print("   2. Sign up for an account")
        print("   3. Get your API key from the dashboard")
        print("\n🔧 Then set the environment variable:")
        print("   export GRADIUM_API_KEY='your-api-key-here'")
        print("\nOr add it to your .env file:")
        print("   echo 'GRADIUM_API_KEY=your-api-key-here' >> .env")
        return False
    
    print(f"✅ GRADIUM_API_KEY found: {api_key[:10]}...")
    
    # Create TTS config
    config = TTSConfig(
        api_key=api_key,
        region="eu",  # or "us"
        voice_id="YTpq7expH9539ERJ",  # Emma voice
    )
    
    tts = TextToSpeech(config)
    
    # Test text
    test_text = "Hello, I am your AI forensic artist assistant. I'm ready to help you."
    
    try:
        print(f"\n🎤 Testing TTS with text: '{test_text}'")
        print(f"📍 Region: {config.region}")
        print(f"🎭 Voice: {config.voice_id} (Emma)")
        print(f"⏳ Connecting to Gradium API...")
        
        audio = await tts.synthesize(test_text)
        
        print(f"\n✅ Success! Synthesized {len(audio)} bytes of audio")
        print(f"   Sample rate: {config.sample_rate} Hz (48kHz)")
        print(f"   Format: PCM 16-bit mono")
        
        # Save to file
        import wave
        output_file = "test_tts_gradium.wav"
        with wave.open(output_file, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)  # 16-bit
            f.setframerate(config.sample_rate)
            f.writeframes(audio)
        
        print(f"💾 Saved audio to: {output_file}")
        print(f"\n🎉 Gradium TTS test passed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TTS test failed: {e}")
        logger.exception("Full error details:")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tts())
    sys.exit(0 if success else 1)
