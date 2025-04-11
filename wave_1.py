import tensorflow as tf
import numpy as np
import librosa
import os
from pydub import AudioSegment
# ✅ تحميل النموذج
model_path = "C:\\Users\\Rania\\Downloads\\model\\modele_wave.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ لم يتم العثور على النموذج في: {model_path}")

model = tf.keras.models.load_model(model_path)
print("✅ النموذج تم تحميله بنجاح!")

# ✅ قائمة المشاعر المدعومة
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']


def preprocess_waveform(waveform, target_length=156544):
    """ تجهيز الموجة الصوتية لتتناسب مع النموذج (طول ثابت) """
    # إذا كانت الموجة الصوتية أطول من الحجم المطلوب، نقوم بتقطيعها
    if len(waveform) > target_length:
        waveform = waveform[:target_length]
    
    # إذا كانت الموجة الصوتية أقصر من الحجم المطلوب، نضيف padding
    elif len(waveform) < target_length:
        padding = target_length - len(waveform)
        waveform = np.pad(waveform, (0, padding), mode='constant')
    
    return np.expand_dims(waveform, axis=0)  # (1, target_length)


def recognize_from_file(file_path, sr=22050, target_length=156544):
    """ التعرف على المشاعر من ملف صوتي """
    try:
        waveform, _ = librosa.load(file_path, sr=sr)
        return recognize_from_waveform(waveform, target_length)
    except Exception as e:
        return f"❌ خطأ أثناء التحميل من الملف: {str(e)}"


def recognize_from_waveform(waveform, target_length=156544):
    """ التعرف على المشاعر من waveform مباشرة """
    try:
        processed_waveform = preprocess_waveform(waveform, target_length)
        prediction = model.predict(processed_waveform)
        emotion_index = np.argmax(prediction)
        emotion = EMOTIONS[emotion_index]
        return emotion
    except Exception as e:
        return f"❌ خطأ أثناء التنبؤ: {str(e)}"


def convert_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"❌ فشل تحويل الملف إلى WAV: {str(e)}")
        return None

