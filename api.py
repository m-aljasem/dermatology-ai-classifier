"""
RESTful API Server for Skin Lesion Classification

This FastAPI server provides endpoints for model predictions and health checks.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

app = FastAPI(
    title="Skin Lesion Classification API",
    description="RESTful API for Skin Lesion Classification predictions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "skin_cancer_model.h5"
STATS_PATH = MODELS_DIR / "None" if stats_filename else None

# Global model
model = None
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
INPUT_SHAPE = (224, 224, 3)

def load_model():
    """Load TensorFlow model."""
    global model
    try:
        if MODEL_PATH.exists():
            model = tf.keras.models.load_model(str(MODEL_PATH))
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️  Model not found at {MODEL_PATH}. API will return errors until model is trained.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Skin Lesion Classification API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Make a prediction from uploaded image."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize and preprocess
        image = image.resize(INPUT_SHAPE[:2])
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        # Predict
        if false:
            # Multi-label prediction
            pred = model.predict(img_array, verbose=0)[0]
            detected = []
            results = {}
            for i, (cls, prob) in enumerate(zip(CLASSES, pred)):
                results[cls] = float(prob)
                if prob > 0.5:
                    detected.append(cls)
            
            return {
                "detected_conditions": detected,
                "probabilities": results,
                "num_conditions": len(detected)
            }
        elif len(CLASSES) == 0:
            # Regression task (bone age)
            pred = model.predict(img_array, verbose=0)[0][0]
            return {
                "prediction": float(pred),
                "unit": "months"
            }
        else:
            # Single-label classification
            pred = model.predict(img_array, verbose=0)
            class_idx = int(np.argmax(pred[0]))
            confidence = float(pred[0][class_idx])
            class_name = CLASSES[class_idx]
            
            probabilities = {cls: float(prob) for cls, prob in zip(CLASSES, pred[0])}
            
            return {
                "prediction": class_name,
                "class_index": class_idx,
                "confidence": confidence,
                "probabilities": probabilities
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "TensorFlow/Keras",
        "input_shape": INPUT_SHAPE,
        "classes": CLASSES,
        "multilabel": false
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
