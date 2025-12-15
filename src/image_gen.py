"""Image generation using FLUX.1-dev from Hugging Face.

Generates forensic sketch images from text prompts.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

# Output directory
GENERATED_DIR = Path(__file__).parent.parent / "generated"
GENERATED_DIR.mkdir(exist_ok=True)


def generate_image(
    prompt: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    width: int = 512,
    height: int = 768,
) -> str:
    """Generate an image using FLUX.1-dev.
    
    Args:
        prompt: Text description for image generation
        output_path: Optional path for output file. If None, auto-generates.
        seed: Random seed for reproducibility
        num_inference_steps: Number of denoising steps (default 28 for FLUX)
        guidance_scale: CFG scale (default 3.5 for FLUX)
        width: Image width
        height: Image height
        
    Returns:
        Path to the generated image
    """
    try:
        import torch
        from diffusers import FluxPipeline
    except ImportError:
        raise ImportError(
            "Install diffusers and torch: pip install diffusers torch accelerate"
        )
    
    # Generate output path if not provided
    if output_path is None:
        image_id = uuid.uuid4().hex[:8]
        output_path = str(GENERATED_DIR / f"sketch_{image_id}.png")
    
    logger.info(f"Loading FLUX.1-dev model...")
    
    # Load the pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    
    # Use GPU if available, otherwise CPU (will be slow)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        pipe = pipe.to(device)
    else:
        # For CPU, enable sequential offloading to reduce memory
        pipe.enable_model_cpu_offload()
    
    # Set seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    # Style the prompt for forensic sketch
    styled_prompt = (
        f"charcoal forensic sketch portrait, black and white, "
        f"police composite drawing style, detailed facial features, "
        f"{prompt}"
    )
    
    logger.info(f"Generating image with prompt: {styled_prompt[:100]}...")
    
    # Generate
    image = pipe(
        prompt=styled_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    
    # Save
    image.save(output_path)
    logger.info(f"Image saved to: {output_path}")
    
    return output_path


def generate_image_api(
    prompt: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """Generate image using Hugging Face Inference API (no local GPU needed).
    
    This is a lighter alternative that uses the HF API instead of local inference.
    Requires HF_TOKEN environment variable.
    
    Args:
        prompt: Text description
        output_path: Optional output path
        seed: Random seed
        
    Returns:
        Path to generated image
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found in .env file or environment")
    
    # Generate output path
    if output_path is None:
        image_id = uuid.uuid4().hex[:8]
        output_path = str(GENERATED_DIR / f"sketch_{image_id}.png")
    
    client = InferenceClient(token=hf_token)
    
    # Style the prompt
    styled_prompt = (
        f"charcoal forensic sketch portrait, black and white, "
        f"police composite drawing style, detailed facial features, "
        f"{prompt}"
    )
    
    logger.info(f"Generating via API: {styled_prompt[:80]}...")
    
    # Generate using API
    image = client.text_to_image(
        prompt=styled_prompt,
        model="black-forest-labs/FLUX.1-dev",
    )
    
    # Save
    image.save(output_path)
    logger.info(f"Image saved to: {output_path}")
    
    return output_path


# Quick test
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="male, oval face, dark skin, green eyes")
    parser.add_argument("--api", action="store_true", help="Use HF API instead of local")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    if args.api:
        path = generate_image_api(args.prompt, seed=args.seed)
    else:
        path = generate_image(args.prompt, seed=args.seed)
    
    print(f"Generated: {path}")
