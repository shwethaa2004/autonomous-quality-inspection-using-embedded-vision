# VisionInspect AI: Autonomous Quality Inspection

This project uses on-device computer vision to classify objects as **GOOD** or **DEFECT**.

## Project Components

1.  **React Frontend (Web App)**: The main interactive interface for data collection, training, and real-time detection directly in your browser using TensorFlow.js.
2.  **Legacy Python Scripts**: Provided for users who prefer a standalone Python/OpenCV environment:
    *   `capture.py`: Dataset collection script.
    *   `train.py`: Model training script using TensorFlow/Keras.
    *   `app.py`: Flask-based inference server.

## Getting Started (Web App)
1.  Allow camera access when prompted.
2.  **Step 1: Collection Mode**. Points the camera at a "GOOD" item and press 'G'. Repeat for "DEFECT" items and press 'D'.
3.  **Step 2: Training**. Click "Train Model" to fine-tune the classifier in your browser.
4.  **Step 3: Detection**. Switch to Detection mode to see real-time status.

## Running Locally (Python)
If you wish to run the Python version:
1.  Install dependencies: `pip install -r requirements.txt`
2.  Collect data: `python capture.py`
3.  Train: `python train.py`
4.  Run Server: `python app.py`
5.  Access at `http://localhost:5000`
