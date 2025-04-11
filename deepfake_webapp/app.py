from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import datetime
import csv

# Import prediction function
from fused_predict import predict_fused_model

app = Flask(__name__, template_folder='templates', static_folder='static')

# Upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Log file
LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'prediction_log.csv')
os.makedirs(LOG_DIR, exist_ok=True)

# Logging function
def log_prediction(filename, result):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [
        timestamp,
        filename,
        result['predicted_label'],
        round(result['video_real_score'], 4),
        round(result['video_fake_score'], 4),
        round(result['audio_real_score'], 4),
        round(result['audio_fake_score'], 4)
    ]
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Filename", "Prediction",
                             "Video_Real", "Video_Fake", "Audio_Real", "Audio_Fake"])
        writer.writerow(row)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        result = predict_fused_model(video_path)
        log_prediction(filename, result)

        if os.path.exists(video_path):
            os.remove(video_path)

        if "error" in result:
            return jsonify({'error': result['error']}), 500

        #return jsonify({'prediction': result['predicted_label'].upper()})
        return jsonify({
            'prediction': result['predicted_label'].upper(),
            'explanation': result['explanation']
        })

@app.route('/logs')
def view_logs():
    if not os.path.exists(LOG_FILE):
        return "No logs available yet."
    with open(LOG_FILE, 'r') as f:
        log_data = f.read().replace('\n', '<br>')
    return f"<div style='padding: 20px; font-family: monospace;'>{log_data}</div>"

if __name__ == '__main__':
    app.run(debug=True)
