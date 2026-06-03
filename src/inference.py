import os
import numpy as np
import tensorflow as tf
import yaml
from preprocessor import load_and_preprocess_image, preprocess_frame

class WasteClassifier:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.classes = self.config['dataset']['classes']
        self.input_shape = tuple(self.config['model']['input_shape'][:2]) # (height, width)
        self.weights_path = self.config['model']['weights_path']
        
        # Load model
        if os.path.exists(self.weights_path):
            print(f"Loading trained weights from {self.weights_path}...")
            self.model = tf.keras.models.load_model(self.weights_path)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model weights not found at {self.weights_path}. Predictions will be unavailable until trained.")
            self.model = None

    def predict_image(self, image_path):
        """
        Runs inference on an image from disk.
        
        Args:
            image_path (str): Absolute or relative path to the image.
            
        Returns:
            dict: Prediction details containing 'class', 'confidence', and 'probabilities'.
        """
        if self.model is None:
            return {"error": "Model weights are not loaded. Train the model first."}
            
        preprocessed = load_and_preprocess_image(image_path, target_size=self.input_shape)
        if preprocessed is None:
            return {"error": f"Failed to load/preprocess image at {image_path}"}
            
        return self._run_inference(preprocessed)

    def predict_frame(self, bgr_frame):
        """
        Runs inference on a numpy frame (BGR format from OpenCV camera or stream).
        
        Args:
            bgr_frame (np.ndarray): In-memory image frame.
            
        Returns:
            dict: Prediction details containing 'class', 'confidence', and 'probabilities'.
        """
        if self.model is None:
            return {"error": "Model weights are not loaded. Train the model first."}
            
        preprocessed = preprocess_frame(bgr_frame, target_size=self.input_shape)
        if preprocessed is None:
            return {"error": "Failed to preprocess camera frame."}
            
        return self._run_inference(preprocessed)

    def _run_inference(self, preprocessed_image):
        # Add batch dimension: shape becomes (1, height, width, 3)
        input_tensor = np.expand_dims(preprocessed_image, axis=0)
        
        # Execute prediction
        predictions = self.model.predict(input_tensor, verbose=0)[0]
        
        # Get highest probability index
        predicted_idx = np.argmax(predictions)
        predicted_class = self.classes[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Format all probabilities
        probabilities = {cls: float(prob) for cls, prob in zip(self.classes, predictions)}
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        }

if __name__ == "__main__":
    # Test classifier init
    classifier = WasteClassifier()
