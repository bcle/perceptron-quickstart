import os
import perceptron

perceptron.configure(
    api_key=os.environ.get("PERCEPTRON_API_KEY"),
    provider="perceptron",
    model="isaac-0.2-2b-preview",
)

image_url = "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/main/cookbook/_shared/assets/capabilities/qna/studio_scene.webp"
image_url = "https://raw.githubusercontent.com/bcle/public-files/main/2018-10-03_2018_Summer_Youth_Olympics_by_Sandro_Halank%E2%80%93001.jpg"
image_url = "https://raw.githubusercontent.com/bcle/public-files/main/people_in_costco_parking_lot.png"

# result = perceptron.question(image_url, "What is in this image?")

result = perceptron.detect(
    image_url,           # str: Local path or URL to image
    classes=["person","car"],   # list[str]: Categories you expect in frame
    expects="box",        # str: "box" | "point" | "polygon"
    reasoning=True        # bool: enable reasoning and include chain-of-thought (when supported)
)

# Access detections (boxes, points, or polygons based on `expects`)
for annotation in result.points or []:
    print(annotation.mention, annotation)

print(result.text)
