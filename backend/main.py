import os
import tempfile
import cv2
import numpy as np
import base64
import sqlite3
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.detector import AdvancedViolationDetector

app = FastAPI(title="Smart Traffic Violation Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Database setup
# ──────────────────────────────────────────────
DB_PATH = "history.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            filename    TEXT NOT NULL,
            media_type  TEXT NOT NULL,
            total_bikes INTEGER NOT NULL DEFAULT 0,
            violations  INTEGER NOT NULL DEFAULT 0,
            safe_riders INTEGER NOT NULL DEFAULT 0,
            details     TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
detector = AdvancedViolationDetector(
    model_path="yolov8n.pt",
    helmet_model_path="runs/detect/helmet_yolov8n/weights/best.pt"
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def save_detection(filename: str, media_type: str, mc_data: list):
    total_bikes = len(mc_data)
    violations  = sum(1 for m in mc_data if m.get("violation", "Safe") != "Safe")
    # Count individual riders who are wearing helmets across all bikes
    safe_riders = sum(
        sum(1 for h in m.get("helmets", []) if h is True)
        for m in mc_data
    )
    conn = get_db()
    conn.execute(
        """INSERT INTO detections (timestamp, filename, media_type, total_bikes, violations, safe_riders, details)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            media_type,
            total_bikes,
            violations,
            safe_riders,
            json.dumps(mc_data),
        ),
    )
    conn.commit()
    conn.close()

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running."}


@app.get("/history")
def get_history(limit: int = 50):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        row_dict = dict(r)
        row_dict["details"] = json.loads(row_dict["details"]) if row_dict["details"] else []
        result.append(row_dict)
    return {"history": result}


@app.delete("/history/clear")
def clear_history():
    conn = get_db()
    conn.execute("DELETE FROM detections")
    conn.commit()
    conn.close()
    return {"message": "History cleared."}


@app.post("/predict")
async def predict_violation(file: UploadFile = File(...)):

    if file.content_type.startswith("video/"):
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_in.write(await file.read())
        temp_in.close()

        cap = cv2.VideoCapture(temp_in.name)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_out.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))

        response_data = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            ann_frame, data = detector.detect(frame)
            out.write(ann_frame)
            if frame_idx % max(int(fps), 1) == 0 and len(data) > 0:
                response_data.extend(data)
            frame_idx += 1

        cap.release()
        out.release()
        os.remove(temp_in.name)

        save_detection(file.filename or "video.mp4", "video", response_data)
        return FileResponse(temp_out.name, media_type="video/mp4", filename="processed.mp4")

    elif file.content_type.startswith("image/"):
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image data."})

        annotated_img, data = detector.detect(image)

        _, buffer = cv2.imencode(".jpg", annotated_img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        save_detection(file.filename or "image.jpg", "image", data)

        return {"motorcycles": data, "image_base64": img_base64}

    return JSONResponse(
        status_code=400,
        content={"error": "Unsupported file type. Use image/jpeg, image/png or video/mp4."},
    )
