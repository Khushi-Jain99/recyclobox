# RecycloBox AI

RecycloBox AI is an intelligent, deep learning-powered waste segregation and classification system designed to categorize waste items into 7 distinct classes: **Plastic**, **Paper**, **Glass**, **Metal**, **Organic**, **Hazardous**, and **Non-Recyclable**.

By using OpenCV for image preprocessing, TensorFlow/Keras for deep learning transfer learning (MobileNetV2), and Streamlit/FastAPI for user interfaces and API endpoints, RecycloBox forms a complete, production-ready computer vision solution. This system is designed for local deployment and integration with IoT-enabled smart dustbins.

---

## 📂 Project Structure

```text
recyclobox/
├── .gitignore                  # Specifies folders and model weights ignored by Git
├── README.md                   # Project overview, setup, and usage guidelines
├── requirements.txt            # Python dependencies (TensorFlow, OpenCV, FastAPI, etc.)
├── config/
│   └── config.yaml             # Configurations for directories, splits, and training
├── data/                       # Local directory for datasets (ignored by Git)
│   ├── raw/                    # Original unzipped folders of dataset images
│   └── processed/              # Stratified split dataset (train, val, test)
├── saved_models/               # Serialized model weights (ignored by Git, directory preserved)
│   └── checkpoints/            # Epoch-by-epoch weights checkpoints
├── notebooks/                  # Notebooks for Jupyter experiments
├── tests/                      # Automated unit tests
│   ├── test_model.py           # Model layer dimension verification
│   └── test_preprocessor.py    # Preprocessing pipeline checks
├── app/                        # Streamlit web application dashboard
│   ├── __init__.py
│   └── app.py                  # Entrypoint for visual dashboard
└── src/                        # Core codebase package
    ├── __init__.py
    ├── preprocessing/          # Dataset splitting, augmentation, and image utilities
    │   ├── __init__.py
    │   ├── preprocessor.py     # OpenCV resizing, conversion, and contrast filters
    │   ├── data_loader.py      # High-performance tf.data loaders
    │   └── prepare_dataset.py  # Script to clean raw folders and split data
    ├── models/                 # Model structures
    │   ├── __init__.py
    │   └── model.py            # CNN and MobileNetV2 architecture definitions
    ├── training/               # Model training scripts
    │   ├── __init__.py
    │   └── train.py            # Feature extraction and fine-tuning execution pipeline
    ├── prediction/             # Inference classes
    │   ├── __init__.py
    │   └── predictor.py        # Wrapper class to run prediction on file/webcam feeds
    ├── api/                    # Web API microservice
    │   ├── __init__.py
    │   └── main.py             # FastAPI prediction endpoints (POST /predict)
    └── utils/                  # Helper utilities
        ├── __init__.py
        └── config_helper.py    # YAML configuration file parser
```

---

## ⚙️ Setup and Installation

### 1. Prerequisite Environments
Ensure you have Python 3.9+ installed.

### 2. Clone and Setup Environment
```bash
git clone <your-repository-url>
cd recyclobox

# Create a virtual environment
python -m venv venv
# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Placing Datasets
Store the dataset subfolders directly under `data/` at the root of the project.
The directory structure inside `data/` should look like this before processing:
- `data/batteries/`
- `data/cans_all_type/`
- `data/ceramic_product/`
- ...

---

## 🚀 Running the System

### 1. Preprocess and Split Dataset
Clean the raw dataset folders and organize them into stratified splits in `data/processed/`:
```bash
python src/preprocessing/prepare_dataset.py
```

### 2. Train the Deep Learning Model
Train the MobileNetV2 transfer learning model (first trains the head, then fine-tunes the top 20 layers of the base model):
```bash
python src/training/train.py
```
*Trained weights will be saved to `saved_models/recyclobox_model.keras` and training history curves will be output to `saved_models/training_curves.png`.*

### 3. Run the Streamlit Dashboard App
Run the interactive user dashboard which supports image upload and live camera feeds:
```bash
streamlit run app/app.py
```

### 4. Run the FastAPI Production API
Run the microservice API server for remote devices to send predictions:
```bash
python src/api/main.py
```
*The server will start at `http://localhost:8000`. You can access documentation at `http://localhost:8000/docs`.*

---

## 🔌 API Documentation Summary

- **`GET /`**: Returns server status.
- **`GET /health`**: Returns model loading status and API health check.
- **`POST /predict`**: Accepts an image file and returns predictions.
  - **Payload**: Multipart file upload (`file`).
  - **Response**:
    ```json
    {
      "class": "plastic",
      "confidence": 0.942,
      "probabilities": {
        "plastic": 0.942,
        "paper": 0.012,
        "glass": 0.005,
        ...
      }
    }
    ```

---

## 🤖 Smart Dustbin Integration

To connect RecycloBox AI with a hardware smart dustbin (utilizing Raspberry Pi, Arduino, or ESP32):

1. **Hardware Setup**: Equip the physical bin with a camera, servo motors for lid control (directing items to different internal compartments), and a status indicator LED.
2. **Network Connection**: Connect the microcontroller/Pi to the local network where the FastAPI server is running.
3. **Capture & Query**: 
   - When an item is placed in the drop zone, trigger the camera to capture a BGR frame.
   - Send the image file via an HTTP POST request to `http://<server-ip>:8000/predict`.
4. **Actuation**:
   - Parse the JSON response for the `"class"` key.
   - Command the servo motors to rotate the sorting funnel to the matching trash bin compartment (e.g. rotate 45 degrees for `metal`, 90 degrees for `plastic`).
