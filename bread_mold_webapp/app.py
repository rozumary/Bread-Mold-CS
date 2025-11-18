import io
import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageDraw
import base64
import tempfile
import numpy as np

def calculate_total_mold_area(boxes):
    """
    Calculate the total area covered by mold boxes, avoiding double counting
    overlapping regions using inclusion-exclusion principle
    """
    if not boxes:
        return 0

    # For simplicity, we'll use a rasterization approach for accurate area calculation
    # Create a binary mask for mold areas
    # Determine the image size from the boxes (or use a reasonable default)
    max_x = max(box[2] for box in boxes) if boxes else 1
    max_y = max(box[3] for box in boxes) if boxes else 1
    
    # Create a mask with sufficient resolution
    mask_width = int(max_x) + 100  # Add padding
    mask_height = int(max_y) + 100

    # Create a binary mask
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # Fill the mask with mold regions
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mask[y1:y2, x1:x2] = 1

    # Calculate the total area by counting non-zero pixels
    total_area = np.count_nonzero(mask)
    return total_area

# === Load local YOLO model (.pt file) ===
MODEL_PATH = "Bread-Mold-CS-main/bread_mold_webapp/my_model.pt"   # <- change to your model filename
model = YOLO(MODEL_PATH)
# ==========================================

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["image"]
    img_bytes = file.read()

    # FIX: create temp file properly for Windows
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(img_bytes)
    temp.close()

    # Run YOLO prediction safely
    results = model.predict(source=temp.name, conf=0.25, iou=0.45)

    # Load image to draw
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    # Collect mold detection boxes for accurate area calculation
    mold_boxes = []
    bread_area = w * h

    detections = results[0].boxes

    for box in detections:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

        color = (255, 0, 0) if "mold" in cls_name.lower() else (0, 120, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 10), f"{cls_name} {conf*100:.2f}%", fill=color)

        if "mold" in cls_name.lower():
            mold_boxes.append((x1, y1, x2, y2))

    # Calculate total mold area without double counting overlapping regions
    mold_area = calculate_total_mold_area(mold_boxes)

    # Cleanup temp file
    os.unlink(temp.name)

    coverage_ratio = mold_area / bread_area
    if coverage_ratio < 0.1:
        risk = "Low"
        action = "Safe to remove moldy part carefully."
    elif coverage_ratio < 0.3:
        risk = "Moderate"
        action = "Do not eat. Dispose bread safely."
    else:
        risk = "Severe"
        action = "Highly contaminated. Dispose immediately."

    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "risk": risk,
        "coverage": round(coverage_ratio * 100, 2),
        "action": action,
        "annotated": f"data:image/jpeg;base64,{encoded_img}"
    })


if __name__ == "__main__":
    app.run(debug=True)
