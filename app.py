from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load model if it exists
model = None
if os.path.exists('model.h5'):
    model = tf.keras.models.load_model('model.h5')
    print("Model loaded.")
else:
    print("Warning: model.h5 not found. Please run train.py first.")

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            if model is not None:
                # Preprocess for MobileNetV2
                img = cv2.resize(frame, (224, 224))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                
                prediction = model.predict(img, verbose=0)
                label = "GOOD" if prediction[0][0] > prediction[0][1] else "DEFECT"
                color = (0, 255, 0) if label == "GOOD" else (0, 0, 255)
                
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
