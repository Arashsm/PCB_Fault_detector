from fastapi import FastAPI, UploadFile, File, Response, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
import shutil
import os
from pathlib import Path
import cv2
from sqlalchemy.orm import Session
import numpy as np

from database import SessionLocal, engine
import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

model_path = "best.pt"  # Update this with the correct path
model = YOLO(model_path)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASSES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper"
]

class Prediction(BaseModel):
    class_name: str
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class PredictResponse(BaseModel):
    filename: str
    predictions: List[Prediction]

def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

def filter_predictions(predictions, conf_limit):
    return [pred for pred in predictions if pred['confidence'] >= conf_limit]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/predict/", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf_limit: float = 0.25,
    db: Session = Depends(get_db)
):
    if not 0 <= conf_limit <= 1:
        raise HTTPException(status_code=400, detail="Confidence limit must be between 0 and 1.")

    file_path = Path(UPLOAD_DIR) / file.filename
    save_upload_file(file, file_path)

    results = model(str(file_path), conf=conf_limit)

    predictions = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x_min, y_min, x_max, y_max = [int(val) for val in box.xyxy[0].cpu().numpy()]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = CLASSES[class_id]
            predictions.append({
                'class_name': class_name,
                'confidence': confidence,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max
            })

    predictions = filter_predictions(predictions, conf_limit)

    for pred in predictions:
        db_prediction = models.PredictionResult(
            filename=file.filename,
            class_name=pred['class_name'],
            confidence=pred['confidence'],
            x_min=pred['x_min'],
            y_min=pred['y_min'],
            x_max=pred['x_max'],
            y_max=pred['y_max']
        )
        db.add(db_prediction)
    db.commit()

    return PredictResponse(filename=file.filename, predictions=predictions)

@app.post("/visualize/")
async def visualize(
    file: UploadFile = File(...),
    conf_limit: float = 0.25
):
    if not 0 <= conf_limit <= 1:
        raise HTTPException(status_code=400, detail="Confidence limit must be between 0 and 1.")

    file_path = Path(UPLOAD_DIR) / file.filename
    save_upload_file(file, file_path)

    results = model(str(file_path), conf=conf_limit)

    image = cv2.imread(str(file_path))

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < conf_limit:
                continue

            x_min, y_min, x_max, y_max = [int(val) for val in box.xyxy[0].cpu().numpy()]
            class_id = int(box.cls[0])
            detections.append({
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'class_id': class_id,
                'confidence': confidence
            })

    for det in detections:
        x_min, y_min, x_max, y_max = det['x_min'], det['y_min'], det['x_max'], det['y_max']
        class_id = det['class_id']
        confidence = det['confidence']
        class_name = CLASSES[class_id]
        label = f"{class_name} ({confidence:.2f})"

        color = (255, 255, 255)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x_min, y_min - text_height - 4), (x_min + text_width, y_min), color, -1)
        cv2.putText(image, label, (x_min, y_min - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()

    return Response(content=img_bytes, media_type="image/png")

@app.get("/")
def read_root():
    return {"message": "PCB Defect Detection API is running."}
