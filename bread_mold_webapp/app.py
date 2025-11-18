import io
import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageDraw
import base64
import tempfile
import torch
from torch.hub import load_state_dict_from_url

# Handle PyTorch 2.6+ security changes for loading models
# Add safe globals for ultralytics models and their dependencies
try:
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.conv import Conv, autopad
    from ultralytics.nn.modules.block import C2f, Bottleneck
    from ultralytics.nn.modules.head import Detect
    torch.serialization.add_safe_globals([DetectionModel, Conv, C2f, Bottleneck, Detect])
except ImportError:
    pass

try:
    # Also add torch.nn modules that might be needed
    from torch.nn.modules.container import Sequential
    from torch.nn.modules.activation import SiLU, Sigmoid
    from torch.nn.modules.pooling import MaxPool2d
    from torch.nn.modules.linear import Linear
    from torch.nn.modules.normalization import BatchNorm2d
    torch.serialization.add_safe_globals([Sequential, SiLU, Sigmoid, MaxPool2d, Linear, BatchNorm2d])
except ImportError:
    pass

# === Load local YOLO model (.pt file) ===
MODEL_PATH = "bread_mold_webapp/my_model.pt"   # <- change to your model filename

# Create a context where we temporarily allow unsafe loading for the model
# This is a workaround for PyTorch 2.6+ security changes
def load_model_with_weights_only_false(model_path):
    original_torch_load = torch.load

    def patched_torch_load(f, map_location=None, **kwargs):
        kwargs['weights_only'] = False # Force weights_only to False
        return original_torch_load(f, map_location=map_location, **kwargs)

    # Temporarily replace torch.load
    torch.load = patched_torch_load
    try:
        model = YOLO(model_path)
    finally:
        # Restore original torch.load
        torch.load = original_torch_load

    return model

model = load_model_with_weights_only_false(MODEL_PATH)

# ==============================================

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
    results = model.predict(source=temp.name, conf=0.40)

    # Load image to draw
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    mold_area = 0
    bread_area = w * h

    detections = results[0].boxes

    # Create a mask to accurately calculate mold coverage without overlapping areas
    mold_mask = Image.new('L', (w, h), 0)
    mask_draw = ImageDraw.Draw(mold_mask)

    for box in detections:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to int for pixel operations

        color = (255, 0, 0) if "mold" in cls_name.lower() else (0, 120, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 10), f"{cls_name} {conf*100:.1f}%", fill=color)

        if "mold" in cls_name.lower():
            # Fill the mold area in the mask to prevent double counting overlapping regions
            mask_draw.rectangle([x1, y1, x2, y2], fill=255)

    # Count the number of pixels in the mold mask to get accurate area
    mold_pixels = sum(mold_mask.getpixel((x, y)) > 0 for x in range(w) for y in range(h))
    mold_area = mold_pixels

    # Cleanup temp file
    os.unlink(temp.name)

    coverage_ratio = min(mold_area / bread_area, 1.0)  # Cap at 100%
    if coverage_ratio == 0:
        risk = "None"
        action = "Safe to eat"
    elif coverage_ratio < 0.1:
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
