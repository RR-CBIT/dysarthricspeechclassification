import os
import uuid
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib
from matplotlib import pyplot as plt
import librosa
import librosa.display
from utils.preprocess import extract_features

app = Flask(__name__)
app.secret_key = 'your-secret-key'

model = tf.keras.models.load_model("model/model.h5")
class_names = ['Normal', 'Dysarthric']

from time import time

@app.context_processor
def inject_time():
    return {'time': time}

UPLOAD_FOLDER = 'static/uploads'
SPECTROGRAM_FOLDER = 'static/spectrograms'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

def generate_spectrogram(file_path, spec_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    # After plotting spectrogram:
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(spec_path)
    plt.close()

@app.route('/', methods=['GET'])
def index():
    error = session.pop('error', None)
    label = session.pop('label', None)
    spec_path = session.pop('spec_path', None)
    return render_template('index.html', label=label, spec_path=spec_path, time=inject_time())

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')

    if not file or file.filename == '':
        session['error'] = "No file selected."
        return redirect(url_for('index'))

    if not file.filename.endswith('.wav'):
        session['error'] = "Invalid file type. Please upload a .wav file."
        return redirect(url_for('index'))

    try:
        file_path = os.path.join(UPLOAD_FOLDER, 'temp.wav')
        file.save(file_path)

        # Generate and store spectrogram
        spec_filename = f"{uuid.uuid4()}.png"
        spec_path = os.path.join(SPECTROGRAM_FOLDER, spec_filename)
        generate_spectrogram(file_path, spec_path)

        # Predict
        features = extract_features(file_path)
        prediction = model.predict(features)
        prob = prediction[0][0]
        print(prob)
        result = class_names[0] if prob <= 0.5 else class_names[1]

        session['label'] = result
        session['spec_path'] = spec_path

        return redirect(url_for('index'))  # PRG: redirect after POST
    except Exception as e:
        print("Error during prediction:", e)

    return redirect(url_for('index'))
if __name__ == '__main__':
    app.run(debug=True)