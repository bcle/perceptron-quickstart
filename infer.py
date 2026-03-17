#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import re
import urllib.request
from pathlib import Path

DEFAULT_MAX_TOKENS = 512
PUBLIC_API_URL = "https://api.perceptron.inc/v1"


def _image_data_uri(image_path: str) -> str:
    img_bytes = Path(image_path).read_bytes()
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    b64 = base64.b64encode(img_bytes).decode()
    return f"data:{mime};base64,{b64}"


def _get(url: str, path: str, api_key: str | None = None) -> dict:
    headers = {"User-Agent": "curl/8.5.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url.rstrip("/") + path, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _post(url: str, body: bytes, api_key: str | None = None) -> dict:
    headers = {"Content-Type": "application/json", "User-Agent": "curl/8.5.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def infer(image_path: str | None, url: str, prompt: str, hint: str | None, max_tokens: int, verbose: bool = False, think: bool = False, api_key: str | None = None, model: str = "isaac-0.2-1b") -> dict:
    content = []
    if hint:
        content.append({"type": "text", "text": f"<hint>{hint.upper()}</hint>"})
    if image_path:
        data_uri = _image_data_uri(image_path)
        content.append({"type": "image_url", "image_url": {"url": data_uri}})
    content.append({"type": "text", "text": prompt})

    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
        "think": think,                                    # ollama extension (ignored by vLLM)
        "chat_template_kwargs": {"enable_thinking": think},  # vLLM convention (ignored by ollama)
    }

    if verbose:
        # Truncate base64 data for readability
        display = json.loads(json.dumps(body))
        for msg in display.get("messages", []):
            for part in msg.get("content", []):
                if part.get("type") == "image_url":
                    url_val = part["image_url"]["url"]
                    if ";base64," in url_val:
                        prefix, data = url_val.split(";base64,", 1)
                        part["image_url"]["url"] = f"{prefix};base64,<{len(data)} chars>"
        print("=== REQUEST ===")
        print(json.dumps(display, indent=2))
        print("=== RESPONSE ===")

    return _post(url, json.dumps(body).encode(), api_key=api_key)


def _parse_boxes(text: str) -> list[dict]:
    """Extract bounding boxes from model output.

    Handles two formats:
      - <collection mention="label"> <point_box> (x1,y1) (x2,y2) </point_box> ... </collection>
      - <point_box mention="label"> (x1,y1) (x2,y2) </point_box>

    Returns list of dicts with keys: label, x1, y1, x2, y2 (all coords 0-1000).
    """
    boxes = []
    coord = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"

    # Collection format: gather boxes under the collection's mention label
    for coll in re.finditer(
        r'<collection\s+mention="([^"]*)">(.*?)</collection>', text, re.DOTALL
    ):
        label = coll.group(1)
        for m in re.finditer(coord + r"\s*" + coord, coll.group(2)):
            boxes.append(dict(label=label, x1=int(m.group(1)), y1=int(m.group(2)),
                              x2=int(m.group(3)), y2=int(m.group(4))))

    # Individual point_box format (skip those already inside a collection)
    coll_spans = {m.span() for m in re.finditer(r'<collection[^>]*>.*?</collection>', text, re.DOTALL)}
    for m in re.finditer(r'<point_box(?:\s+mention="([^"]*)")?>\s*' + coord + r'\s*' + coord, text):
        # Skip if inside a collection
        if any(start <= m.start() < end for start, end in coll_spans):
            continue
        label = m.group(1) or ""
        boxes.append(dict(label=label, x1=int(m.group(2)), y1=int(m.group(3)),
                          x2=int(m.group(4)), y2=int(m.group(5))))

    return boxes


def _draw_boxes(image_path: str, boxes: list[dict], output_path: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    palette = ["#FF4444", "#44AAFF", "#44FF88", "#FFB844", "#CC44FF", "#FF44CC"]
    label_colors: dict[str, str] = {}

    for box in boxes:
        label = box["label"]
        if label not in label_colors:
            label_colors[label] = palette[len(label_colors) % len(palette)]
        color = label_colors[label]

        x1 = int(box["x1"] / 1000 * w)
        y1 = int(box["y1"] / 1000 * h)
        x2 = int(box["x2"] / 1000 * w)
        y2 = int(box["y2"] / 1000 * h)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if label:
            draw.text((x1 + 2, max(y1 - 14, 0)), label, fill=color)

    img.save(output_path)
    print(f"Saved annotated image to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Send an image to the Isaac inference server")
    parser.add_argument("image", nargs="?", help="Path to the image file")
    parser.add_argument("--no-image", action="store_true", help="Send only the prompt, no image")
    url_group = parser.add_mutually_exclusive_group()
    url_group.add_argument("--url", default="http://localhost:8091/v1", help="Server base URL")
    url_group.add_argument("--public-api", action="store_true", help=f"Use the public API endpoint ({PUBLIC_API_URL})")
#   parser.add_argument("--prompt", default="find people and vehicles")
    parser.add_argument("--prompt", default="Detect all people and vehicles in this image.")
    parser.add_argument("--hint", default="box", help="Structured output hint (box/point/polygon), or empty to disable")
    parser.add_argument("--model", default="isaac-0.2-1b", help="Model to use for inference")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--think", action="store_true", help="Enable chain-of-thought reasoning (disabled by default)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Also print the JSON request before the response")
    parser.add_argument("-o", "--output", metavar="FILE", help="Draw bounding boxes on the image and save to FILE")
    parser.add_argument("--contentonly", action="store_true", help="Print only the content field of the response")
    parser.add_argument("--api-key", default=None, help="API key (overrides PERCEPTRON_API_KEY env var)")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    url = PUBLIC_API_URL if args.public_api else args.url
    api_key = args.api_key or os.environ.get("PERCEPTRON_API_KEY")

    if args.list_models:
        data = _get(url, "/models", api_key=api_key)
        for m in data.get("data", []):
            print(m["id"])
        return

    if args.no_image:
        image_path = None
    else:
        if not args.image:
            parser.error("image is required unless --no-image is set")
        image_path = args.image

    hint_default_set = args.hint == parser.get_default("hint")
    hint = (args.hint or None) if (image_path or not hint_default_set) else None
    result = infer(image_path, url, args.prompt, hint, args.max_tokens, verbose=args.verbose, think=args.think, api_key=api_key, model=args.model)

    if args.contentonly:
        print(result["choices"][0]["message"]["content"])
    else:
        print(json.dumps(result, indent=2))

    if args.output and image_path:
        content = result["choices"][0]["message"]["content"]
        boxes = _parse_boxes(content)
        if boxes:
            _draw_boxes(image_path, boxes, args.output)
        else:
            print("No bounding boxes found in response.")


if __name__ == "__main__":
    main()
