import io
import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageDraw
import base64
import tempfile

# === Load local YOLO model (.pt file) ===
MODEL_PATH = "C:\\Users\\ASUS\\Bread-Mold-CS-2\\Bread-Mold-CS-main\\bread_mold_webapp\\my_model.pt"   # <- change to your model filename
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
    results = model.predict(source=temp.name, conf=0.40)

    # Load image to draw
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    mold_area = 0
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
        draw.text((x1, y1 - 10), f"{cls_name} {conf*100:.1f}%", fill=color)

        if "mold" in cls_name.lower():
            mold_area += (x2 - x1) * (y2 - y1)

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
