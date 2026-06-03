import os
import sys
import io
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory of api folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from prediction.predictor import WasteClassifier

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
    classifier = WasteClassifier(config_path="config/config.yaml")

@app.get("/")
def read_root():
    return {
        "project": "RecycloBox AI",
        "status": "Online",
        "description": "FastAPI waste classification service. POST an image to /predict to classify."
    }

@app.get("/health")
def health_check():
    if classifier is not None and classifier.model is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "degraded", "model_loaded": False, "detail": "Model weights are missing or not loaded."}

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
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
            
        # Run prediction
        result = classifier.predict_frame(cv_image)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
