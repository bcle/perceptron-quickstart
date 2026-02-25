from pathlib import Path
from urllib.request import urlretrieve
import os
from perceptron import configure, detect
from PIL import Image, ImageDraw

configure(
    provider="perceptron",
    model="isaac-0.2-2b-preview",
    api_key=os.environ.get("PERCEPTRON_API_KEY"),
)

# Download reference frame
IMAGE_URL = "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/main/cookbook/_shared/assets/capabilities/detection/ppe_line.webp"
IMAGE_PATH = Path("ppe_line.webp")
ANNOTATED_PATH = Path("ppe_line_annotated.png")

if not IMAGE_PATH.exists():
    urlretrieve(IMAGE_URL, IMAGE_PATH)

# Detect PPE
result = detect(
    str(IMAGE_PATH),
    classes=["helmet", "vest"],
)

print(result.text)
print(f"Detections: {len(result.points or [])}")

# Draw detections
img = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(img)
pixel_boxes = result.points_to_pixels(width=img.width, height=img.height) or []

for box in pixel_boxes:
    draw.rectangle(
        [
            int(box.top_left.x),
            int(box.top_left.y),
            int(box.bottom_right.x),
            int(box.bottom_right.y),
        ],
        outline="lime",
        width=3,
    )
    label = box.mention or "target"
    confidence = getattr(box, "confidence", None)
    if confidence is not None:
        label = f"{label} ({confidence:.2f})"
    draw.text((int(box.top_left.x), max(int(box.top_left.y) - 18, 0)), label, fill="lime")

img.save(ANNOTATED_PATH)
print(f"Saved annotated frame to {ANNOTATED_PATH}")