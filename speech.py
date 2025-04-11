import librosa
import numpy as np
import tensorflow as tf
import os

# ✅ تحميل النموذج المدرب
model_path = "C:\\Users\\Rania\\Downloads\\model\\modeleee.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ لم يتم العثور على النموذج في: {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
    print("✅ النموذج تم تحميله بنجاح!")
except Exception as e:
    raise RuntimeError(f"❌ فشل في تحميل النموذج: {str(e)}")

# ✅ قائمة المشاعر المدعومة
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

def extract_features(file_path, target_time_steps=182, n_mfcc=13):  # ← رفعنا عدد الـ MFCC لأننا نحتاج وسطهم
    """
    استخراج الميزات الصوتية باستخدام MFCC وتحويلها إلى شكل يناسب النموذج.
    """
    try:
        data, sr = librosa.load(file_path, sr=22050, duration=2.5, offset=0.6)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc).T  # (?, n_mfcc)

        if mfcc.shape[0] > target_time_steps:
            mfcc = mfcc[:target_time_steps, :]
        else:
            pad = target_time_steps - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant')

        # ✅ أخذ القناة اللي في المنتصف بدل القناة الأولى
        mid_channel_index = mfcc.shape[1] // 2
        mfcc = mfcc[:, mid_channel_index:mid_channel_index+1]  # (182, 1)

        # ✅ إعادة التشكيل إلى (1, 182, 1)
        mfcc = np.expand_dims(mfcc, axis=0)

        return mfcc

    except Exception as e:
        print(f"❌ خطأ في استخراج الميزات: {str(e)}")
        return None

def recognizee(file_path):
    """
    التنبؤ بالعاطفة من ملف صوتي باستخدام النموذج المدرب.
    """
    try:
        features = extract_features(file_path)
        if features is None:
            return "❌ خطأ في استخراج الميزات!"

        print("🔹 شكل البيانات المدخلة للنموذج:", features.shape)
        prediction = model.predict(features)
        emotion_index = np.argmax(prediction)
        emotion = EMOTIONS[emotion_index]

        return f" {emotion}"

    except Exception as e:
        return f"❌ خطأ أثناء التنبؤ: {str(e)}"
