import json
import os
import uuid
from typing import Any, Dict, List

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "models", "best.onnx")
CLASSES_PATH = os.path.join(APP_DIR, "models", "classes.json")
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

app = FastAPI(title="Dental AI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = None
class_names: Dict[str, str] = {}


def load_classes() -> Dict[str, str]:
    if not os.path.exists(CLASSES_PATH):
        return {"0": "Implant", "1": "Fillings", "2": "Impacted Tooth", "3": "Cavity"}
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_session():
    global session, class_names
    if session is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Put best.onnx in models/best.onnx")
        class_names = load_classes()
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return session


def letterbox(image: np.ndarray, new_shape: int = 640):
    h, w = image.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    dw, dh = (new_shape - nw) // 2, (new_shape - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, scale, dw, dh


def preprocess(image_bytes: bytes):
    pil_img = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    original = np.array(pil_img)
    img, scale, dw, dh = letterbox(original, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, original.shape[1], original.shape[0], scale, dw, dh


def xywh_to_xyxy(x, y, w, h):
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def nms(dets: List[Dict[str, Any]], iou_thres: float):
    dets = sorted(dets, key=lambda x: x["confidence"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if d["class_id"] != best["class_id"] or iou(d["box"], best["box"]) < iou_thres]
    return keep


def postprocess(output, orig_w, orig_h, scale, dw, dh):
    # YOLOv8 ONNX output usually: (1, 4 + nc, 8400)
    pred = output[0]
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    detections = []
    for row in pred:
        scores = row[4:]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])
        if conf < CONF_THRESHOLD:
            continue

        x, y, w, h = row[:4]
        box = xywh_to_xyxy(float(x), float(y), float(w), float(h))

        # remove letterbox padding and scale back to original image
        box[0] = (box[0] - dw) / scale
        box[1] = (box[1] - dh) / scale
        box[2] = (box[2] - dw) / scale
        box[3] = (box[3] - dh) / scale

        box[0] = max(0, min(orig_w, box[0]))
        box[1] = max(0, min(orig_h, box[1]))
        box[2] = max(0, min(orig_w, box[2]))
        box[3] = max(0, min(orig_h, box[3]))

        label = class_names.get(str(class_id), f"class_{class_id}")
        detections.append({
            "label": label,
            "class_id": class_id,
            "confidence": round(conf, 4),
            "box": [round(v, 2) for v in box]
        })

    return nms(detections, IOU_THRESHOLD)


@app.get("/")
def root():
    return {"status": "ok", "message": "Dental AI Service is running"}


@app.get("/health")
def health():
    model_exists = os.path.exists(MODEL_PATH)
    return {
        "status": "ok",
        "model_exists": model_exists,
        "model_path": "models/best.onnx",
        "model_version": "dental-yolo-onnx-v1",
        "classes": load_classes()
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    patient_id: str = Form(None),
    doctor_id: str = Form(None)
):
    try:
        sess = get_session()
        image_bytes = await image.read()
        input_tensor, orig_w, orig_h, scale, dw, dh = preprocess(image_bytes)
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: input_tensor})
        findings = postprocess(outputs[0], orig_w, orig_h, scale, dw, dh)

        if findings:
            labels = sorted(set(f["label"] for f in findings))
            summary = "Detected: " + ", ".join(labels)
            confidence = round(max(f["confidence"] for f in findings), 4)
        else:
            summary = "No clear dental findings detected."
            confidence = 0.0

        return {
            "status": "success",
            "summary": summary,
            "findings": findings,
            "confidence": confidence,
            "heatmap_url": None,
            "model_version": "dental-yolo-onnx-v1",
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "request_id": str(uuid.uuid4())
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "summary": "AI analysis failed.",
                "findings": [],
                "confidence": 0,
                "heatmap_url": None,
                "model_version": "dental-yolo-onnx-v1",
                "error": str(e)
            }
        )
