import shutil
import time
import traceback
import zipfile
import joblib
import pandas as pd 
import wave
import librosa
import threading
import matplotlib.pyplot as plt # type: ignore
import librosa.display
import seaborn as sns # type: ignore
import numpy as np
from flask import Flask, Response, render_template, request, jsonify,send_file, url_for
import os
from sklearn.calibration import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import soundfile as sf
from keras.src.saving import load_model
import tensorflow as tf
from werkzeug.utils import secure_filename, send_from_directory,safe_join
from delete_files import delete
from speech import recognizee
from speech import extract_features
from wave_1 import recognize_from_file , recognize_from_waveform
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input , BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.callbacks import ModelCheckpoint
from flask_cors import CORS
from tensorflow.keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import sys
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')  # إضافة هذا السطر قبل استيراد pyplot
import matplotlib.pyplot as plt
import sounddevice as sd

ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
CORS(app) 
app.config['MAX_CONTENT_LENGTH'] = 10000 * 1024 * 1024  # 600MB

# تحديد المجلدات
UPLOAD_FOLDER = "uploads"
EXTRACT_FOLDER = "extracted_data"
RECORDINGS_FOLDER = os.path.join(UPLOAD_FOLDER, 'recordings')
TEST_FOLDER = os.path.join(os.getcwd(), 'test')  # مجلد الاختبار

# إنشاء المجلدات إذا لم تكن موجودة
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# إضافة المجلدات إلى إعدادات التطبيق
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECORDINGS_FOLDER'] = RECORDINGS_FOLDER
app.config['EXTRACT_FOLDER'] = EXTRACT_FOLDER
app.config['TEST_FOLDER'] = TEST_FOLDER  # ✅ تأكد من أن هذا السطر يُنفذ قبل أي استخدام له


# المتغيرات العامة
training_progress = 0  # نسبة التقدم
training_done = False  # هل اكتمل التدريب؟

# تحميل النموذج
MODEL_PATH = "C:/Users/Rania/Downloads/model/model.h5"
try:
    model = load_model(MODEL_PATH)
    print("✅ النموذج تم تحميله بنجاح!")
    print("✅ شكل إدخال النموذج:", model.input_shape)
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {e}")
# أضف هذا في بداية ملف app.py بعد الاستيرادات
emotions = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
emotion_map = {emotion.lower(): idx for idx, emotion in enumerate(emotions)}
# تصنيفات العواطف
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
EMOTION_MAP = {"a": "Anger", "d": "Disgust", "f": "Fear", "h": "Happiness", "n": "Neutral", "s": "Sadness", "su": "Surprise"}
label_encoder = LabelEncoder()
label_encoder.fit(EMOTIONS)
def extract_emotion_from_filename(filename):
    filename = filename.lower()
    if filename.startswith("su"): return EMOTION_MAP["su"]
    return EMOTION_MAP.get(filename[0], "Unknown")

# 🔹 التحقق من أن الملف المرفوع بصيغة صحيحة
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def convert_to_wav(input_path, output_path):
    try:
        print(f"🔄 تحويل {input_path} إلى WAV...")
        
        # 📌 تحميل الملف باستخدام pydub
        audio = AudioSegment.from_file(input_path)
        
        # 📌 ضبط الصوت على قناة واحدة ومعدل 22050 هرتز
        audio = audio.set_channels(1).set_frame_rate(22050)

        # 📌 تصدير الملف بصيغة WAV
        audio.export(output_path, format="wav")
        
        print(f"✅ تم التحويل إلى WAV: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ فشل التحويل إلى WAV: {e}")
        return None

@app.route('/')
def home():
    return render_template('home.html')
@app.route("/test")
def test_page():
    return render_template("test.html")
@app.route("/train")
def train_page():
    return render_template("train.html")
@app.route('/uploadvoise')
def uploadvoise():
    return render_template('uploadvoise.html')
@app.route('/fileprede')
def fileprede():
    return render_template('fileprede.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/main')
def main_page():
    return render_template('main.html')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    return render_template('search.html')
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    try:
        upload_folder = os.path.abspath(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(upload_folder, filename)

        if not os.path.exists(file_path):
            return "❌ الملف غير موجود!", 404

        return send_file(file_path, as_attachment=True)  # استخدام send_file بدلاً من send_from_directory
    except Exception as e:
        print(f"❌ خطأ أثناء جلب الملف: {e}")
        return "❌ خطأ أثناء تحميل الملف", 500

@app.route('/result', methods=['GET', 'POST'])
def upload_file():
    delete()  # حذف الملفات القديمة

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("result.html", name="No file part!")

        f = request.files['file']

        if f.filename == '':
            return render_template("result.html", name="No file selected!")

        if not allowed_file(f.filename):
            return render_template("result.html", name="Invalid file type! Only .wav is allowed.")

        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)  # حفظ الملف

        ans = recognizee(file_path)  # تمرير الملف إلى `recognize()`

        return render_template("result.html", name="File uploaded successfully!", ans=ans)
    
@app.route('/upload_files', methods=['POST'])
def upload_audio_files():
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
    print("🔹 Received files:", request.files)

    if 'files' not in request.files:
        print("❌ No files uploaded!")
        return jsonify({"error": "No files uploaded!"}), 400

    files = request.files.getlist('files')  # جلب جميع الملفات المرفوعة
    results = []  # قائمة لتخزين نتائج كل ملف

    for file in files:
        if file.filename == '':
            print("❌ Empty filename, skipping...")
            continue

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"✅ File saved at {file_path}")

        # 🔹 تمرير الملف إلى دالة recognize للحصول على التنبؤ مباشرة
        predicted_emotion = recognizee(file_path)
        print(f"🎯 Predicted Emotion for {filename}: {predicted_emotion}")

        # 🔹 تخزين النتيجة
        results.append({"filename": filename, "emotion": predicted_emotion})

    if not results:
        return jsonify({"error": "No valid files processed"}), 400

    return jsonify({"results": results})


@app.route("/record", methods=["POST"])
def record_audio():
    """
    استقبال الصوت من المستخدم، معالجته، والتنبؤ بالعاطفة.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "❌ لا يوجد إدخال صوتي!"}), 400

        # ✅ استقبال الملف الصوتي
        audio_file = request.files["file"]
        filename = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(filename)
        print(f"✅ تم حفظ الملف في: {filename}")

        # ✅ تحويل الصوت إلى WAV إذا لم يكن بالفعل
        fixed_path = os.path.join(UPLOAD_FOLDER, "fixed_audio.wav")
        converted_path = convert_to_wav(filename, fixed_path)

        if not converted_path:
            return jsonify({"error": "❌ فشل تحويل الملف إلى WAV!"}), 500

        # ✅ تحليل المشاعر باستخدام `recognize()`
        print("🔍 جارٍ تحليل الصوت...")
        predicted_emotion = recognizee(fixed_path)

        if "❌" in predicted_emotion:
            return jsonify({"error": predicted_emotion}), 500

        print(f"🎯 التوقع النهائي: {predicted_emotion}")

        return jsonify({"emotion": predicted_emotion})

    except Exception as e:
        print(f"❌ خطأ أثناء تحليل الصوت: {e}")
        return jsonify({"error": f"❌ خطأ أثناء تحليل الصوت: {str(e)}"}), 500

# ✅ دالة تحميل البيانات الصوتية وتحويلها إلى ميزات
@app.route("/list_subfolders", methods=["POST"])
def list_subfolders():
    data = request.json
    base_path = data.get("data_dir", "").replace("\\", "/")

    if not base_path or not os.path.exists(base_path):
        return jsonify({"error": "⚠️ المسار غير موجود أو غير صالح"}), 400

    # قراءة جميع المجلدات الفرعية داخل `Train`
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    if not subfolders:
        return jsonify({"error": "⚠️ لا توجد مجلدات فرعية صالحة"}), 400
    
    return jsonify({"subfolders": subfolders, "message": "✅ تم العثور على المجلدات الفرعية"}), 200

def extract_dataset(zip_path):
    """ استخراج البيانات والتأكد من أن المسار النهائي صحيح """
    
    # 🔥 حذف أي نسخة سابقة من المجلد
    if os.path.exists(EXTRACT_FOLDER):
        print(f"🗑️ حذف المجلد القديم: {EXTRACT_FOLDER}")
        shutil.rmtree(EXTRACT_FOLDER)

    os.makedirs(EXTRACT_FOLDER, exist_ok=True)

    # ✅ فك ضغط الملفات داخل extracted_data
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)

    print("✅ تم فك الضغط بنجاح!")

    # ✅ تتبع المجلدات بعد فك الضغط
    print("📂 محتوى المجلد بعد فك الضغط:", os.listdir(EXTRACT_FOLDER))

    # ✅ البحث عن المجلد الصحيح تلقائيًا
    corrected_path = find_first_valid_folder(EXTRACT_FOLDER)

    if not os.listdir(corrected_path):
        raise ValueError(f"⚠️ المجلد {corrected_path} لا يحتوي على أي ملفات!")

    print(f"📂 المسار النهائي بعد التصحيح: {corrected_path}")

    return corrected_path

def find_first_valid_folder(base_path):
    """ البحث عن أول مجلد داخلي يحتوي على ملفات """
    
    all_items = os.listdir(base_path)
    if not all_items:
        raise ValueError(f"🚨 لم يتم العثور على أي بيانات داخل {base_path}")

    # ✅ إذا كانت الملفات موجودة مباشرة في extracted_data نعيد هذا المسار
    for item in all_items:
        item_path = os.path.join(base_path, item)
        if os.path.isfile(item_path):  
            return base_path.replace("\\", "/")  # استخراج المسار الصحيح

    # ✅ البحث عن أول مجلد داخلي يحتوي على بيانات
    for item in all_items:
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and os.listdir(item_path):  # المجلد يحتوي على ملفات
            return item_path.replace("\\", "/")

    raise ValueError("🚨 لم يتم العثور على أي ملفات داخل المجلد المستخرج!")

# ✅ دالة تحميل البيانات الصوتية وتحويلها إلى ميزات
def load_data(data_dir, is_test=False):
    emotions = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
    emotion_map = {e.lower(): i for i, e in enumerate(emotions)}
    X, y = [], []

    if not os.path.exists(data_dir):
        print(f"❌ خطأ: المجلد {data_dir} غير موجود!")
        return np.array([]), np.array([])

    if is_test:
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if file.endswith('.wav'):
                try:
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(None)
                except Exception as e:
                    print(f"⚠️ خطأ أثناء معالجة الملف {file_path}: {e}")
    else:
        for emotion in emotions:
            emotion_path = os.path.join(data_dir, emotion)
            if not os.path.exists(emotion_path):
                print(f"⚠️ المجلد غير موجود: {emotion_path}")
                continue

            for file in os.listdir(emotion_path):
                file_path = os.path.join(emotion_path, file)
                if file.endswith('.wav'):
                    try:
                        features = extract_features(file_path)
                        if features is not None:
                            X.append(features)
                            y.append(emotion_map[emotion.lower()])
                    except Exception as e:
                        print(f"⚠️ خطأ أثناء معالجة الملف {file_path}: {e}")

    if not X:
        print("❌ لم يتم العثور على أي بيانات صالحة!")
        return np.array([]), np.array([])

    X = np.array(X)
    X = np.expand_dims(X, axis=-1)
    y = np.array(y)

    return X, y

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def pitch(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)

def extract_features(data, sr):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc))
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    return np.hstack((result, mel))

def get_features(path):
    data, sr = librosa.load(path, duration=2.5, offset=0.6)
    features = []
    
    # الإصدار الأصلي
    features.append(extract_features(data, sr))
    
    # مع الضوضاء
    noise_data = noise(data)
    features.append(extract_features(noise_data, sr))
    
    # مع التمديد وتغيير النغمة
    stretched = stretch(data)
    pitched = pitch(stretched, sr)
    features.append(extract_features(pitched, sr))
    
    return np.array(features)

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"error": "❌ لا يوجد ملف مرفوع!"}), 400

    file = request.files['file']
    if not file.filename.endswith('.zip'):
        return jsonify({"error": "❌ الملف يجب أن يكون بصيغة ZIP!"}), 400

    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    try:
        # طباعة المسار والتأكد من الحفظ
        print(f"📥 يتم حفظ الملف في: {zip_path}")  
        file.save(zip_path)
        print("✅ تم حفظ الملف بنجاح!")

        # فك الضغط
        extract_path = extract_dataset(zip_path)

        # التأكد من أن المجلد الناتج غير فارغ
        if not os.listdir(extract_path):
            raise ValueError(f"⚠️ المجلد {extract_path} لا يحتوي على أي ملفات!")

        print(f"📤 يتم إرسال المسار إلى الواجهة: {extract_path}")  # ✅ تتبع المسار المرسل

        return jsonify({"folder_path": extract_path})

    except Exception as e:
        return jsonify({"error": f"❌ حدث خطأ أثناء فك الضغط: {str(e)}"}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        data_dir = data.get('data_dir')
        epochs = data.get('epochs', 50)
        
        if not data_dir or not os.path.exists(data_dir):
            return jsonify({"error": "مسار البيانات غير صحيح!"}), 400
        
        X, Y = [], []
        for emotion_dir in os.listdir(data_dir):
            dir_path = os.path.join(data_dir, emotion_dir)
            if not os.path.isdir(dir_path):
                continue
            
            # تحويل اسم المجلد إلى مؤشر رقمي
            emotion_idx = emotion_map.get(emotion_dir.lower())
            if emotion_idx is None:
                continue  # تخطى المجلدات غير المعروفة
            
            for file in os.listdir(dir_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(dir_path, file)
                    features = get_features(file_path)
                    for f in features:
                        X.append(f)
                        Y.append(emotion_idx)
        
        if len(X) == 0:
            return jsonify({"error": "لا توجد بيانات صالحة للتدريب!"}), 400
        
        # تحضير البيانات
        encoder = OneHotEncoder()
        Y_encoded = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
        
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(X), Y_encoded, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        

        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(encoder, "encoder.pkl")
        # بناء النموذج
        model = Sequential()
        model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
        model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

        model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
        model.add(Dropout(0.2))

        model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

        model.add(Flatten())

# إضافة L2 regularization للطبقة الكثيفة
        model.add(Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))

# طبقة الإخراج
        model.add(Dense(units=7, activation='softmax'))

# تجميع النموذج
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=64
        )
        
        return jsonify({
            "accuracy": history.history['accuracy'],
            "val_accuracy": history.history['val_accuracy'],
            "loss": history.history['loss'],
            "val_loss": history.history['val_loss']
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/progress')
def extract_featuress(data, sr):
    result = np.array([])
    try:
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result = np.hstack((result, zcr))

        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma_stft))

        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfcc))

        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms))

        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr, n_mels=108).T, axis=0)
        result = np.hstack((result, mel))

        return result[:162]  # تعديل: إرجاع الشكل (162،) بدلاً من (162، 1)

    except Exception as e:
        print(f"❌ خطأ في استخراج الميزات: {e}")
        return np.zeros(162)  # تعديل: إرجاع مصفوفة 1D بالحجم 162

def get_featuress(path):
    try:
        data, sr = librosa.load(path, duration=2.5, offset=0.6)
        features = []

        features.append(extract_featuress(data, sr))

        noise_data = noise(data)
        features.append(extract_featuress(noise_data, sr))

        stretched = stretch(data)
        pitched = pitch(stretched, sr)
        features.append(extract_featuress(pitched, sr))

        return np.mean(np.array(features), axis=0)  # تعديل: لا حاجة لإعادة تشكيله إلى (162، 1)

    except Exception as e:
        print(f"❌ خطأ في تحميل الملف {path}: {e}")
        return np.zeros(162)  # تعديل: إرجاع مصفوفة 1D بالحجم 162
def test_modele(audio_files):
    features_list = [get_featuress(file) for file in audio_files]
    X_test = np.array(features_list)  # الآن X_test سيكون (num_samples, 162)

    # تطبيق StandardScaler
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)  # ضبط القيم ضمن نطاق مناسب للنموذج
    X_test = np.expand_dims(X_test, axis=-1)  # إذا كان النموذج يحتاج بعدًا إضافيًا

    # توقع العواطف
    predictions = model.predict(X_test)
    predicted_emotions = [EMOTIONS[np.argmax(pred)] for pred in predictions]

    return predicted_emotions

# تجربة النموذج على مجلد معين
audio_folder = "C:\\Users\\Rania\\Downloads\\datasat\\Validation"
audio_files = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith(".wav")]

if audio_files:
    results = test_modele(audio_files)
    for file, emotion in zip(audio_files, results):
        print(f"🎵 الملف: {os.path.basename(file)}  --> 🎭 العاطفة المتوقعة: {emotion}")
else:
    print("⚠️ لم يتم العثور على أي ملفات صوتية في المجلد!")


@app.route('/test', methods=['GET', 'POST'])
def test_model():
    if request.method == 'GET':
        return render_template('test.html')
    
    elif request.method == 'POST':
        try:
            # التحقق من تحميل الملف
            if 'test_dataset' not in request.files:
                return jsonify({'status': 'Error', 'message': '❌ No test dataset uploaded.'})
            
            test_file = request.files['test_dataset']
            if test_file.filename == '':
                return jsonify({'status': 'Error', 'message': '❌ No file selected!'})

            # حفظ الملف
            filename = secure_filename(test_file.filename)
            filepath = os.path.join(app.config['TEST_FOLDER'], filename)
            test_file.save(filepath)
            print(f"✅ Test dataset saved at {filepath}.")

            # التحقق مما إذا كان الملف ZIP
            if not zipfile.is_zipfile(filepath):
                return jsonify({'status': 'Error', 'message': '❌ The uploaded file is not a valid ZIP archive.'})

            # استخراج الملفات
            extract_path = os.path.join(app.config['TEST_FOLDER'], 'test_extracted')
            os.makedirs(extract_path, exist_ok=True)

            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"✅ Test dataset extracted to {extract_path}.")

            # جمع الملفات الصوتية والتصنيفات
            audio_files = []
            labels = []
            for root, _, files in os.walk(extract_path):
                for file in files:
                    if file.endswith(".wav"):
                        audio_files.append(os.path.join(root, file))
                        labels.append(os.path.basename(root))  # اسم المجلد هو التصنيف

            if len(audio_files) < 2:
                return jsonify({'status': 'Error', 'message': '❌ The test dataset must contain at least 2 classes.'})

            print(f"✅ Found {len(audio_files)} audio files.")

            # تحميل النموذج
            MODEL_PATH = "C:/Users/Rania/Downloads/model/modelee.h5"
            if not os.path.exists(MODEL_PATH):
                return jsonify({'status': 'Error', 'message': f'❌ Model not found at: {MODEL_PATH}'})

            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully.")

            # استخراج الميزات
            # استخراج الميزات
            features_list = [get_featuress(file) for file in audio_files]
            X_test = np.array(features_list)

# **إصلاح مشكلة StandardScaler مع البيانات ثلاثية الأبعاد**
            if len(X_test.shape) == 3:  
                X_test = X_test.reshape(X_test.shape[0], -1)  # تحويل البيانات إلى بعدين

            scaler = StandardScaler()
            X_test = scaler.fit_transform(X_test)

# إعادة تشكيل البيانات لتمريرها إلى النموذج
            X_test = np.expand_dims(X_test, axis=-1)


            # ترميز التصنيفات
            encoder = OneHotEncoder()
            y_test_encoded = encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()

            # تقييم النموذج
            evaluation = model.evaluate(X_test, y_test_encoded, verbose=1)
            evaluation_accuracy = evaluation[1] * 100
            print(f"✅ Model Evaluation Accuracy: {evaluation_accuracy:.2f}%")

            # التنبؤات
            y_pred = np.argmax(model.predict(X_test), axis=1)
            y_true = np.argmax(y_test_encoded, axis=1)

            # إنشاء مصفوفة الارتباك
            conf_matrix = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=list(set(labels)), output_dict=True)

            # حفظ مصفوفة الارتباك
            conf_matrix_path = os.path.join(app.static_folder, 'test', 'confusion_matrix.png')
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(set(labels)), yticklabels=list(set(labels)))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(conf_matrix_path)
            plt.close()

            return jsonify({
                'status': 'Test Completed',
                'accuracy': evaluation_accuracy,
                'confusion_matrix': url_for('static', filename='test/confusion_matrix.png'),
                'classification_report': report
            })

        except Exception as e:
            print("❌ Error occurred:", traceback.format_exc())
            return jsonify({'status': 'Error', 'message': str(e)})
@app.route('/progress')
def progress():
    def generate():
        for i in range(101):  # إرسال التقدم من 0 إلى 100
            yield f"data: {{\"progress\": {i}}}\n\n"
            time.sleep(0.1)  # محاكاة عملية تستغرق وقتاً
    return Response(generate(), content_type='text/event-stream')
   
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
