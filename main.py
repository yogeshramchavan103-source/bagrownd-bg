import io
import os
import threading
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/u2netp.onnx")

# Maximum uploaded file size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# Maximum image dimension after decoding
MAX_IMAGE_DIMENSION = 2048

# U2NETP input resolution
MODEL_SIZE = 320

# Only one inference at a time.
INFERENCE_LOCK = threading.Lock()

app = FastAPI(
    title="U2NETP Background Removal API",
    version="1.0.0"
)

session: Optional[ort.InferenceSession] = None
input_name: Optional[str] = None


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

def load_model():
    global session, input_name

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found: {MODEL_PATH}"
        )

    options = ort.SessionOptions()

    # Keep CPU/RAM usage controlled.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    # Sequential execution generally uses less memory.
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Graph optimization.
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name


@app.on_event("startup")
def startup_event():
    load_model()


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

def preprocess(image: Image.Image):
    original_size = image.size

    # Prevent extremely large images from consuming excessive RAM.
    if max(original_size) > MAX_IMAGE_DIMENSION:
        image.thumbnail(
            (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
            Image.Resampling.LANCZOS
        )

    resized = image.resize(
        (MODEL_SIZE, MODEL_SIZE),
        Image.Resampling.BILINEAR
    )

    # float32 is required by the ONNX model.
    data = np.asarray(resized, dtype=np.float32) / 255.0

    # HWC -> CHW -> NCHW
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)

    return data, image.size


# ---------------------------------------------------------
# Mask normalization
# ---------------------------------------------------------

def normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.squeeze(mask).astype(np.float32)

    minimum = float(mask.min())
    maximum = float(mask.max())

    difference = maximum - minimum

    if difference < 1e-8:
        return np.zeros_like(mask, dtype=np.uint8)

    mask = (mask - minimum) / difference
    mask = np.clip(mask * 255.0, 0, 255)

    return mask.astype(np.uint8)


# ---------------------------------------------------------
# Background removal
# ---------------------------------------------------------

def remove_background(image_bytes: bytes) -> bytes:
    if session is None or input_name is None:
        raise RuntimeError("Model is not loaded")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()

        # Re-open after verify().
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image."
        )

    try:
        model_input, output_size = preprocess(image)

        # Only one U2NETP inference at a time.
        with INFERENCE_LOCK:
            outputs = session.run(
                None,
                {input_name: model_input}
            )

        # First output is the saliency/mask output.
        mask = normalize_mask(outputs[0])

        mask_image = Image.fromarray(mask, mode="L")

        mask_image = mask_image.resize(
            output_size,
            Image.Resampling.BILINEAR
        )

        # Put mask into alpha channel.
        image.putalpha(mask_image)

        # Encode directly in memory.
        output = io.BytesIO()

        image.save(
            output,
            format="PNG",
            optimize=False
        )

        return output.getvalue()

    finally:
        # Release references as soon as possible.
        try:
            del image
        except Exception:
            pass


# ---------------------------------------------------------
# API endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "U2NETP Background Removal API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "u2netp"
    }


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    # Basic content-type validation.
    if file.content_type:
        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp",
            "image/tiff"
        }

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=415,
                detail="Unsupported image type."
            )

    data = await file.read(MAX_FILE_SIZE + 1)

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB."
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Empty image file."
        )

    result = remove_background(data)

    return Response(
        content=result,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="removed-bg.png"',
            "Cache-Control": "no-store"
        }
    )


# ---------------------------------------------------------
# Local execution
# ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1
    )
