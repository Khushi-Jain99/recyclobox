import os
import sys
import io
import numpy as np
from PIL import Image
import cv2
import random
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Add parent directory of api folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from prediction.predictor import WasteClassifier, get_mock_prediction

app = FastAPI(
    title="RecycloBox AI API",
    description="Microservice for real-time deep learning waste image segregation and classification.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None

@app.on_event("startup")
def load_model():
    global classifier
    # Expect config/config.yaml at the root workspace directory
    # If starting api from the root, paths should align.
    classifier = WasteClassifier(config_path="backend/config/config.yaml")

# Serve the UI static files at /static
# Make sure frontend static folder exists
os.makedirs("frontend/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = "frontend/index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(
            content="""
            <html>
                <head><title>RecycloBox AI</title></head>
                <body style="font-family: sans-serif; text-align: center; padding: 50px; background: #0d0e15; color: #fff;">
                    <h1>RecycloBox AI API is Online</h1>
                    <p>Frontend template is loading/missing. Please create <code>frontend/index.html</code>.</p>
                </body>
            </html>
            """,
            status_code=200
        )

@app.get("/health")
def health_check():
    if classifier is not None and classifier.model is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "degraded", "model_loaded": False, "detail": "Model weights are missing or not loaded. Running in Simulator mode."}

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...), mock: bool = Query(False)):
    # Validate file type
    content_type = file.content_type
    filename = file.filename or ""
    is_image = False
    if content_type and content_type.startswith("image/"):
        is_image = True
    elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
        is_image = True
        
    if not is_image:
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    # Check if we should fall back to mock predictions
    model_loaded = classifier is not None and classifier.model is not None
    classes = classifier.classes if classifier else None
    if mock or not model_loaded:
        return get_mock_prediction(file.filename, classes=classes)
        
    try:
        # Read file bytes
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert PIL to OpenCV (numpy array in BGR format)
        cv_image = np.array(pil_image)
        if len(cv_image.shape) == 3: # Color image
            # PIL is RGB, OpenCV is BGR
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        else:
            raise HTTPException(status_code=400, detail="Image must be in RGB/Color mode.")
            
        # Run prediction using the real ML model
        result = classifier.predict_frame(cv_image)
        
        if "error" in result:
            # If real inference fails, fall back to mock prediction
            print(f"ML Inference error: {result['error']}. Falling back to mock prediction.")
            return get_mock_prediction(file.filename, classes=classes)
            
        result["is_mock"] = False
        return result
        
    except Exception as e:
        # Fall back to mock prediction on other errors
        print(f"Process error: {str(e)}. Falling back to mock prediction.")
        return get_mock_prediction(file.filename, classes=classes)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

