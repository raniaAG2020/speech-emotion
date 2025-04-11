from flask import Flask, request, jsonify, render_template,send_from_directory
import os
import librosa
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

import pandas as pd

from sklearn.preprocessing import LabelEncoder

import traceback

from utils256 import extract_features_all
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# تحميل النموذج
MODEL_PATH = "C:\\Users\\Rania\\Downloads\\datasat\\k256_model.h5"

# تحميل النموذج
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# طباعة ملخص النموذج
print(model.summary())

try:
    model = load_model(MODEL_PATH)
    print("✅ النموذج تم تحميله بنجاح!")
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {e}")



# تصنيفات العواطف
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']


def test_model_manually(file_path):
    """دالة لاختبار نموذج التعرف على العواطف يدويًا عبر تمرير ملف صوتي"""
    try:
        # استخراج الميزات الصوتية
        mfcc_features = extract_features_all(file_path)
        if mfcc_features is None:
            print("❌ فشل استخراج الميزات الصوتية")
            return None

        print(f"🔹 شكل الميزات: {mfcc_features.shape}")

        # تنفيذ التوقع باستخدام النموذج
        predictions = model.predict(mfcc_features)
        predicted_emotion = EMOTIONS[np.argmax(predictions)]
        predictions = model.predict( mfcc_features)
        print("🔹 Raw Predictions:", predictions)

        print(f"🎯 العاطفة المتوقعة: {predicted_emotion}")
        return predicted_emotion

    except Exception as e:
        print("❌ خطأ أثناء التنبؤ:", str(e))
        return None


# اختبار يدوي - أدخل مسار ملف صوتي هنا
file_path = "C:/Users/Rania/Downloads/datasat/voisTest/f404.wav"  # استبدليه بمسار ملف صوتي حقيقي
test_model_manually(file_path)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم اختيار أي ملف"}), 400

    files = request.files.getlist('file')
    saved_files = []

    for file in files:
        if file.filename == '':
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        saved_files.append(filename)

    return jsonify({"message": "تم رفع الملفات بنجاح", "files": saved_files})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# في ملف app.py (الجزء الخاص بـ Flask)
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "النموذج غير محمل"}), 500

    file = request.files.get('file')
    if not file:
        return jsonify({"error": "لم يتم رفع أي ملف"}), 400

    try:
        # حفظ الملف مؤقتًا
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

        # استخراج الميزات
        features = extract_features_all(filepath)
        
        # التحقق من الشكل المتوقع (1, 162, 256)
        if features.shape != (1, 162, 256):
            print(f"❌ خطأ: شكل المدخلات {features.shape} غير متطابق مع (1, 162, 256)")
            return jsonify({"error": "شكل الميزات غير صحيح"}), 500

        print(f"✅ الشكل النهائي للميزات: {features.shape}")

        # التنبؤ
        predictions = model.predict(features)
        predicted_label = np.argmax(predictions)
        predicted_emotion = EMOTIONS[predicted_label]

        return jsonify({"predicted_emotion": predicted_emotion})

    except Exception as e:
        print(f"❌ خطأ أثناء التنبؤ: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
        app.run(debug=True)
