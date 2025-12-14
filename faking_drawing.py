"""Utilities for generating a staged forensic-sketch reveal animation."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import requests

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency
    st = None


def _load_image(image_source: str) -> np.ndarray:
    """Load an image from a local path or remote URL."""

    source_path = Path(image_source)
    if source_path.exists():
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Failed to read image from {image_source}")
        return image

    if not image_source.startswith(("http://", "https://")):
        raise ValueError(f"Path not found and not a URL: {image_source}")

    response = requests.get(image_source, timeout=15)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes from URL")
    return image


def process_image_for_3_stage_reveal(image_source: str) -> Tuple[str, str, str]:
    """Return base64 layers (construction, charcoal, final) for a given source."""

    original_img = _load_image(image_source)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_img, 100, 200)
    edges_inverted = 255 - edges
    stage_1_img = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR)

    inverted_img = 255 - gray_img
    blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), 0)
    inverted_blurred_img = 255 - blurred_img
    stage_2_img = cv2.divide(gray_img, inverted_blurred_img, scale=256.0)
    stage_2_img = cv2.cvtColor(stage_2_img, cv2.COLOR_GRAY2BGR)

    def encode_to_b64(img):
        _, buffer = cv2.imencode(".png", img)
        return base64.b64encode(buffer).decode("utf-8")

    return (
        encode_to_b64(stage_1_img),
        encode_to_b64(stage_2_img),
        encode_to_b64(original_img),
    )


CARD_TEMPLATE = """
<style>
    .canvas-container {{
        position: relative;
        width: {width}px;
        height: {height}px;
        background-color: #fff;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        margin: 0 auto;
        border-radius: 4px;
        overflow: hidden;
    }}

    .layer {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}

    #layer1 {{
        opacity: 1;
        clip-path: polygon(0 0, 0 0, 0 100%, 0% 100%);
        animation: wipe-diagonal 2s forwards ease-out;
    }}

    #layer2 {{
        opacity: 0;
        animation: simple-fade 2s forwards ease-in-out;
        animation-delay: 1.8s;
    }}

    #layer3 {{
        opacity: 0;
        animation: simple-fade 1.5s forwards ease-in;
        animation-delay: 3.5s;
    }}

    @keyframes wipe-diagonal {{
        0% {{ clip-path: polygon(0 0, 0 0, 0 100%, 0% 100%); }}
        100% {{ clip-path: polygon(0 0, 150% 0, 100% 150%, 0% 150%); }}
    }}

    @keyframes simple-fade {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}
</style>

<div class="canvas-container">
    <img id="layer1" class="layer" src="data:image/png;base64,{stage1}" />
    <img id="layer2" class="layer" src="data:image/png;base64,{stage2}" />
    <img id="layer3" class="layer" src="data:image/png;base64,{stage3}" />
</div>
"""


def build_animation_html(stage1: str, stage2: str, stage3: str, *, width: int = 500, height: int = 500) -> str:
    """Return the HTML snippet containing layered images and CSS."""

    return CARD_TEMPLATE.format(stage1=stage1, stage2=stage2, stage3=stage3, width=width, height=height)


def write_preview_html(image_source: str, output_file: Path, *, width: int = 500, height: int = 500) -> Path:
    """Generate a standalone HTML preview for a given image source."""

    stage1, stage2, stage3 = process_image_for_3_stage_reveal(image_source)
    html_snippet = build_animation_html(stage1, stage2, stage3, width=width, height=height)
    full_html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Sketch Reveal</title></head><body>{html_snippet}</body></html>"""
    output_file.write_text(full_html, encoding="utf-8")
    return output_file


def show_complex_drawing_animation(image_source: str, *, width: int = 500, height: int = 500) -> None:
    """Render the animation inside Streamlit (if available)."""

    if st is None:  # pragma: no cover - guard for optional dependency
        raise RuntimeError("Streamlit is not installed. Use write_preview_html for local testing.")

    with st.spinner("Compiling forensic layers..."):
        s1, s2, final = process_image_for_3_stage_reveal(image_source)

    html_code = build_animation_html(s1, s2, final, width=width, height=height)
    st.components.v1.html(html_code, height=height + 10)

    if st.button("Replay Animation"):
        st.audio("/home/akpale/hackatons/prep_first_hackaton_ever/sketching_sfx.mp3", autoplay=True)
        show_complex_drawing_animation(image_source, width=width, height=height)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a staged reveal HTML preview for testing.")
    parser.add_argument("image_source", help="Path or URL to the image you want to animate.")
    parser.add_argument("--output", type=Path, default=Path("sketch_reveal_preview.html"), help="HTML file to write.")
    parser.add_argument("--width", type=int, default=500, help="Canvas width in pixels.")
    parser.add_argument("--height", type=int, default=500, help="Canvas height in pixels.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        path = write_preview_html(args.image_source, args.output, width=args.width, height=args.height)
    except Exception as exc:
        parser.error(str(exc))

    print(f"Preview saved to {path.resolve()}")


if __name__ == "__main__":
    main()