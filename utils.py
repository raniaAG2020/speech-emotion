import librosa
import numpy as np
import tensorflow as tf
import os

# ✅ تحميل النموذج المدرب
model_path = "C:\\Users\\Rania\\Downloads\\model\\model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ لم يتم العثور على النموذج في: {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
    print("✅ النموذج تم تحميله بنجاح!")
except Exception as e:
    raise RuntimeError(f"❌ فشل في تحميل النموذج: {str(e)}")

# ✅ قائمة المشاعر المدعومة
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

def extract_features(file_path, target_time_steps=162, n_mfcc=40):
    """
    استخراج الميزات الصوتية باستخدام MFCC وتحويلها إلى شكل يناسب النموذج.
    """
    try:
        # تحميل الصوت
        data, sr = librosa.load(file_path, sr=22050, duration=2.5, offset=0.6)
        
        # استخراج الميزات MFCC
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc).T

        # تطبيع الطول الزمني
        if mfcc.shape[0] > target_time_steps:
            mfcc = mfcc[:target_time_steps, :]
        else:
            pad = target_time_steps - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant')

        # ✅ إعادة تشكيل الميزات لتناسب النموذج (1, 162, 40)
        mfcc = np.expand_dims(mfcc, axis=-1)  # (162, 40, 1)
        return np.expand_dims(mfcc, axis=0)  # (1, 162, 40, 1)
    
    except Exception as e:
        print(f"❌ خطأ في استخراج الميزات: {str(e)}")
        return None
