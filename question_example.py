from pathlib import Path
from urllib.request import urlretrieve
import os
from perceptron import configure, question
from PIL import Image, ImageDraw

configure(
  provider="perceptron",
  model="isaac-0.2-2b-preview",
  api_key=os.environ.get("PERCEPTRON_API_KEY"),
)

# Download reference image
IMAGE_URL = "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/main/cookbook/_shared/assets/capabilities/qna/studio_scene.webp"
IMAGE_PATH = Path("studio_scene.webp")
ANNOTATED_PATH = Path("studio_scene_annotated.png")

if not IMAGE_PATH.exists():
  urlretrieve(IMAGE_URL, IMAGE_PATH)

# Ask a grounded question
prompt = "What stands out in this studio scene? Call out props or people with boxes."
result = question(
  str(IMAGE_PATH),
  prompt,
  expects="box",
)

print(result.text)

# Draw citations
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
    outline="cyan",
    width=3,
  )
  label = box.mention or "answer"
  confidence = getattr(box, "confidence", None)
  if confidence is not None:
    label = f"{label} ({confidence:.2f})"
  draw.text((int(box.top_left.x), max(int(box.top_left.y) - 18, 0)), label, fill="cyan")

img.save(ANNOTATED_PATH)
print(f"Saved annotated image to {ANNOTATED_PATH}")