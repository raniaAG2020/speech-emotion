from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask_cors import CORS
from utils import extract_features_all
from delete_files import delete
from speech import recognize

UPLOAD_FOLDER = 'uploads'
RECORDINGS_FOLDER = os.path.join(UPLOAD_FOLDER, 'recordings')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECORDINGS_FOLDER'] = RECORDINGS_FOLDER

# تحميل النموذج
MODEL_PATH = "C:/Users/Rania/Downloads/model/model.h5"
try:
    model = load_model(MODEL_PATH)
    print("✅ النموذج تم تحميله بنجاح!")
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {e}")

# تصنيفات العواطف
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
EMOTION_MAP = {"a": "Anger", "d": "Disgust", "f": "Fear", "h": "Happiness", "n": "Neutral", "s": "Sadness",
               "su": "Surprise"}


def extract_emotion_from_filename(filename):
    filename = filename.lower()
    if filename.startswith("su"): return EMOTION_MAP["su"]
    return EMOTION_MAP.get(filename[0], "Unknown")


@app.route('/')
def main():
    return render_template('main.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    return render_template('search.html')


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    delete()  # حذف الملفات القديمة
    if 'files' not in request.files:
        return jsonify({"error": "لم يتم اختيار أي ملفات"}), 400

    files = request.files.getlist('files')
    saved_files = []
    for file in files:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        saved_files.append(filename)

    return jsonify({"message": "تم رفع الملفات بنجاح", "files": saved_files})


@app.route('/predict', methods=['POST'])
def predict():
    print("🔹 Received request at /predict")
    print("🔹 Received files:", request.files)  # ➜ طباعة الملفات المستلمة

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    print("✅ File received:", file.filename)

    # حفظ الملف في مجلد مؤقت
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # استخراج الميزات وتحليل العاطفة
    mfcc_features = extract_features_all(file_path)
    if mfcc_features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    print("📊 Extracted Features Shape:", mfcc_features.shape)

    preds = model.predict(mfcc_features)
    predicted_emotion = EMOTIONS[np.argmax(preds)]

    return jsonify({"emotion": predicted_emotion})

@app.route('/record', methods=['POST'])
def record_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        print("✅ File received:", file.filename)  # ➜ تأكيد استقبال الملف
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['RECORDINGS_FOLDER'], filename)
    os.makedirs(app.config['RECORDINGS_FOLDER'], exist_ok=True)
    file.save(file_path)

    mfcc_features = extract_features_all(file_path)
    if mfcc_features is None:
        return jsonify({"error": "Failed to extract features"}), 400

    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    preds = model.predict(mfcc_features)
    predicted_emotion = EMOTIONS[np.argmax(preds)]
    os.remove(file_path)

    return jsonify({"filename": filename, "emotion": predicted_emotion})


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
