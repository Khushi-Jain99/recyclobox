import os
import sys
import numpy as np
import tensorflow as tf
import random

# Add parent directory of prediction folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.preprocessor import load_and_preprocess_image, preprocess_frame
from utils.config_helper import load_config

class WasteClassifier:
    def __init__(self, config_path="config/config.yaml"):
        self.config = load_config(config_path)
        if not self.config:
            raise ValueError(f"Could not load configuration from {config_path}")
            
        self.classes = self.config['dataset']['classes']
        self.input_shape = tuple(self.config['model']['input_shape'][:2])
        self.weights_path = self.config['model']['weights_path']
        
        if os.path.exists(self.weights_path):
            print(f"Loading trained weights from {self.weights_path}...")
            self.model = tf.keras.models.load_model(self.weights_path)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model weights not found at {self.weights_path}. Predictions unavailable.")
            self.model = None

    def predict_image(self, image_path):
        if self.model is None:
            return {"error": "Model weights are not loaded. Train the model first."}
            
        preprocessed = load_and_preprocess_image(image_path, target_size=self.input_shape)
        if preprocessed is None:
            return {"error": f"Failed to load/preprocess image at {image_path}"}
            
        return self._run_inference(preprocessed)

    def predict_frame(self, bgr_frame):
        if self.model is None:
            return {"error": "Model weights are not loaded. Train the model first."}
            
        preprocessed = preprocess_frame(bgr_frame, target_size=self.input_shape)
        if preprocessed is None:
            return {"error": "Failed to preprocess camera frame."}
            
        return self._run_inference(preprocessed)

    def _run_inference(self, preprocessed_image):
        input_tensor = np.expand_dims(preprocessed_image, axis=0)
        predictions = self.model.predict(input_tensor, verbose=0)[0]
        
        predicted_idx = np.argmax(predictions)
        predicted_class = self.classes[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        probabilities = {cls: float(prob) for cls, prob in zip(self.classes, predictions)}
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        }

def get_mock_prediction(filename: str, classes=None):
    filename = filename.lower()
    if classes is None:
        classes = ["plastic", "paper", "glass", "metal", "organic", "hazardous", "non_recyclable"]
    
    # Check for keyword matches in filename
    detected_class = None
    if "plastic" in filename or "bag" in filename or "bottle" in filename:
        detected_class = "plastic"
    elif "paper" in filename or "cardboard" in filename or "newspaper" in filename:
        detected_class = "paper"
    elif "glass" in filename or "cup" in filename or "container" in filename:
        detected_class = "glass"
    elif "metal" in filename or "can" in filename or "tin" in filename or "aluminium" in filename:
        detected_class = "metal"
    elif "organic" in filename or "food" in filename or "scrap" in filename or "leaf" in filename or "yard" in filename or "banana" in filename or "apple" in filename or "vegetable" in filename:
        detected_class = "organic"
    elif "hazardous" in filename or "battery" in filename or "paint" in filename or "pesticide" in filename or "e-waste" in filename or "bulb" in filename:
        detected_class = "hazardous"
    elif "non_recyclable" in filename or "ceramic" in filename or "diaper" in filename or "napkin" in filename or "styrofoam" in filename or "trash" in filename:
        detected_class = "non_recyclable"
        
    # If no keyword found, let's randomly pick a class
    if detected_class is None:
        detected_class = random.choice(classes)
        
    # Generate probabilities
    confidence = round(random.uniform(0.85, 0.97), 3)
    remaining = 1.0 - confidence
    
    # Distribute remaining to other classes
    num_others = len(classes) - 1
    if num_others > 0:
        other_probs = [random.uniform(0.01, remaining) for _ in range(num_others)]
        sum_others = sum(other_probs)
        if sum_others > 0:
            other_probs = [round((p / sum_others) * remaining, 3) for p in other_probs]
        else:
            other_probs = [round(remaining / num_others, 3)] * num_others
            
        # Adjust last one so sum is exactly 1.0
        diff = round(1.0 - (confidence + sum(other_probs)), 3)
        other_probs[-1] = round(other_probs[-1] + diff, 3)
    else:
        other_probs = []
        
    probabilities = {}
    other_idx = 0
    for cls in classes:
        if cls == detected_class:
            probabilities[cls] = confidence
        else:
            probabilities[cls] = max(0.0, other_probs[other_idx])
            other_idx += 1
            
    return {
        "class": detected_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "is_mock": True
    }
