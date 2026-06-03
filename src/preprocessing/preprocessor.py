import cv2
import numpy as np

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image from path using OpenCV, resizes, converts to RGB, and normalizes it.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Failed to load image at {image_path}")
            return None
        return preprocess_frame(img, target_size)
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocesses an in-memory frame (e.g., from webcams or uploads) using OpenCV.
    """
    if frame is None or frame.size == 0:
        return None
        
    img_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    return img_normalized

def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance contrast.
    """
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
