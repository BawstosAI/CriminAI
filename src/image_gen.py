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
DEMO_DIR = GENERATED_DIR / "demo"
DEMO_DIR.mkdir(exist_ok=True)


def get_demo_image_path() -> Optional[str]:
    """Return the first demo image path if available."""
    if not DEMO_DIR.exists() or not DEMO_DIR.is_dir():
        return None

    image_exts = (".png", ".jpg", ".jpeg", ".webp")
    demo_images = sorted(
        [p for p in DEMO_DIR.iterdir() if p.suffix.lower() in image_exts and p.is_file()]
    )
    if not demo_images:
        return None

    demo_path = str(demo_images[0])
    logger.info("Using demo image instead of generation: %s", demo_path)
    return demo_path


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


def generate_image_sdxl_turbo(
    prompt: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    num_inference_steps: int = 3,
    guidance_scale: float = 0.0,
) -> str:
    """Generate an image using SDXL-Turbo locally."""
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        raise ImportError("Install diffusers and torch: pip install diffusers torch accelerate")

    if output_path is None:
        image_id = uuid.uuid4().hex[:8]
        output_path = str(GENERATED_DIR / f"sketch_{image_id}.png")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info(f"Loading SDXL-Turbo model on {device} dtype={dtype}")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    )
    pipe.to(device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    styled_prompt = (
        f"charcoal forensic sketch portrait, black and white, "
        f"police composite drawing style, detailed facial features, "
        f"{prompt}"
    )

    logger.info(f"Generating via SDXL-Turbo: {styled_prompt[:80]}...")
    image = pipe(
        prompt=styled_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    image.save(output_path)
    logger.info(f"Image saved to: {output_path}")
    return output_path


def generate_image_api(
    prompt: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """Generate image using HF API; fall back to local SDXL-Turbo if unavailable."""
    # Generate output path
    if output_path is None:
        image_id = uuid.uuid4().hex[:8]
        output_path = str(GENERATED_DIR / f"sketch_{image_id}.png")

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        logger.warning("huggingface_hub not installed; falling back to SDXL-Turbo")
        return generate_image_sdxl_turbo(prompt, output_path=output_path, seed=seed)
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN missing; falling back to SDXL-Turbo")
        return generate_image_sdxl_turbo(prompt, output_path=output_path, seed=seed)
    
    client = InferenceClient(token=hf_token)
    
    styled_prompt = (
        f"charcoal forensic sketch portrait, black and white, "
        f"police composite drawing style, detailed facial features, "
        f"{prompt}"
    )
    
    logger.info(f"Generating via API: {styled_prompt[:80]}...")
    
    try:
        image = client.text_to_image(
            prompt=styled_prompt,
            model="black-forest-labs/FLUX.1-dev",
        )
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"HF API failed ({e}); falling back to SDXL-Turbo")
        return generate_image_sdxl_turbo(prompt, output_path=output_path, seed=seed)


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
