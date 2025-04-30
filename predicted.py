# import pickle
# import numpy as np
# import librosa
# import os
# from collections import Counter
# from sklearn.preprocessing import StandardScaler

# # Load trained models
# modeldir = r"D:\code\python\intership\main\emotion\models"
# models = {
#     "SVM": os.path.join(modeldir, "mfcc_savee_trained-model.svm"),
#     "MLP": os.path.join(modeldir, "mfcc_savee_trained-model.mlp"),
#     "Logistic Regression": os.path.join(modeldir, "mfcc_savee_trained-model.reg"),
#     "Decision Tree": os.path.join(modeldir, "mfcc_savee_trained-model.dt"),
#     "Random Forest": os.path.join(modeldir, "mfcc_savee_trained-model.rf"),
#     "KNN": os.path.join(modeldir, "mfcc_savee_trained-model.knn")
# }

# # Emotion Mapping
# emotion_mapping = {
#     0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
#     4: "Neutral", 5: "Sad", 6: "Pleasant Surprised"
# }

# # Load Feature Scaler
# with open(os.path.join(modeldir, "scaler.pkl"), 'rb') as f:
#     scaler = pickle.load(f)

# # Feature Extraction Function
# # Feature Extraction (Fixed to Use 13 Features)
# def extract_features(audio_path, n_mfcc=13):
#     y, sr = librosa.load(audio_path, sr=None)
#     mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
#     return mfccs.reshape(1, -1)  # Ensure shape matches trained models

# # Predict Emotion (Fixed for 13 Features)
# def predict_emotion(audio_path):
#     features = extract_features(audio_path)
#     features = scaler.transform(features)  # Apply the same scaling

#     predictions = []

#     for model_name, model_path in models.items():
#         try:
#             with open(model_path, 'rb') as file:
#                 model = pickle.load(file)
#             pred = model.predict(features)[0]
#             predictions.append(pred)
#             #print(f"{model_name} Prediction: {emotion_mapping.get(int(pred), 'Unknown')}")
#         except Exception as e:
#             print(f"Error using {model_name}: {e}")

#     # Majority Voting
#     #final_prediction = Counter(predictions).most_common(1)[0][0]
#     final_prediction = predictions[0]
#     print(f"\nFinal Predicted Emotion: {emotion_mapping.get(int(final_prediction), 'Unknown')}")

# Example Usage
# audio_file = r"D:\code\python\intership\main\frontend\emotion\evolution\disgust\YAF_whip_disgust.wav"
# predict_emotion(audio_file)

# Uncomment the following lines to test with raw audio data
# import pickle
# import numpy as np
# import librosa
# import os
# from collections import Counter
# from sklearn.preprocessing import StandardScaler
# import io

# # Load trained models
# modeldir = r"D:\code\python\intership\main\emotion\models"
# models = {
#     "SVM": os.path.join(modeldir, "mfcc_savee_trained-model.svm"),
#     "MLP": os.path.join(modeldir, "mfcc_savee_trained-model.mlp"),
#     "Logistic Regression": os.path.join(modeldir, "mfcc_savee_trained-model.reg"),
#     "Decision Tree": os.path.join(modeldir, "mfcc_savee_trained-model.dt"),
#     "Random Forest": os.path.join(modeldir, "mfcc_savee_trained-model.rf"),
#     "KNN": os.path.join(modeldir, "mfcc_savee_trained-model.knn")
# }

# # Emotion Mapping
# emotion_mapping = {
#     0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
#     4: "Neutral", 5: "Sad", 6: "Pleasant Surprised"
# }

# # Load Feature Scaler
# with open(os.path.join(modeldir, "scaler.pkl"), 'rb') as f:
#     scaler = pickle.load(f)

# # Feature Extraction Function
# def extract_features(audio_data, n_mfcc=13):
#     y, sr = librosa.load(io.BytesIO(audio_data), sr=None)
#     mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
#     return mfccs.reshape(1, -1)  # Ensure shape matches trained models

# # Predict Emotion (Taking Raw Audio Data)
# def predict_emotion_from_audio(audio_data):
#     features = extract_features(audio_data)
#     features = scaler.transform(features)  # Apply the same scaling

#     predictions = []

#     for model_name, model_path in models.items():
#         try:
#             with open(model_path, 'rb') as file:
#                 model = pickle.load(file)
#             pred = model.predict(features)[0]
#             predictions.append(pred)
#         except Exception as e:
#             print(f"Error using {model_name}: {e}")

#     # Majority Voting
#     final_prediction = predictions[0]  # Using the first model for simplicity
#     emotion = emotion_mapping.get(int(final_prediction), 'Unknown')
#     print(f"\nFinal Predicted Emotion: {emotion}")
#     return emotion




# import pickle
# import numpy as np
# import librosa
# import soundfile as sf
# import os
# import io
# from collections import Counter
# from sklearn.preprocessing import StandardScaler

# # Load trained models
# modeldir = r"D:\code\python\intership\main\emotion\models"
# models = {
#     "SVM": os.path.join(modeldir, "mfcc_savee_trained-model.svm"),
#     "MLP": os.path.join(modeldir, "mfcc_savee_trained-model.mlp"),
#     "Logistic Regression": os.path.join(modeldir, "mfcc_savee_trained-model.reg"),
#     "Decision Tree": os.path.join(modeldir, "mfcc_savee_trained-model.dt"),
#     "Random Forest": os.path.join(modeldir, "mfcc_savee_trained-model.rf"),
#     "KNN": os.path.join(modeldir, "mfcc_savee_trained-model.knn")
# }

# # Emotion Mapping
# emotion_mapping = {
#     0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
#     4: "Neutral", 5: "Sad", 6: "Pleasant Surprised"
# }

# # Load the scaler used during model training
# with open(os.path.join(modeldir, "scaler.pkl"), 'rb') as f:
#     scaler = pickle.load(f)

# # Feature extraction from in-memory audio bytes
# def extract_features(audio_data, n_mfcc=13):
#     y, sr = sf.read(io.BytesIO(audio_data))  # Load audio bytes
#     if y.ndim > 1:
#         y = np.mean(y, axis=1)  # Convert stereo to mono
#     mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
#     return mfccs.reshape(1, -1)

# # Predict emotion using all models and return the majority vote
# def predict_emotion_from_audio(audio_data):
#     features = extract_features(audio_data)
#     features = scaler.transform(features)

#     predictions = []
#     model_votes = {}

#     for model_name, model_path in models.items():
#         try:
#             with open(model_path, 'rb') as file:
#                 model = pickle.load(file)
#             pred = model.predict(features)[0]
#             predictions.append(pred)
#             model_votes[model_name] = emotion_mapping.get(int(pred), 'Unknown')
#         except Exception as e:
#             print(f"❌ Error using {model_name}: {e}")

#     # Display individual model predictions (optional)
#     # print("\n🎯 Individual Model Predictions:")
#     for model_name, emotion in model_votes.items():
#         print(f" - {model_name}: {emotion}")

#     # Get the majority vote
#     final_prediction = Counter(predictions).most_common(1)[0][0]
#     final_emotion = emotion_mapping.get(int(final_prediction), 'Unknown')

#     # print(f"\n✅ Final Predicted Emotion (Majority Vote): {final_emotion}")
#     return final_emotion

# # Optional test from file
# if __name__ == "__main__":
#     audio_path = r"D:\code\python\intership\main\frontend\YAF_wire_disgust.wav"
#     with open(audio_path, 'rb') as f:
#         audio_bytes = f.read()
#     predict_emotion_from_audio(audio_bytes)

import pickle
import numpy as np
import librosa
import os
import io
import soundfile as sf
from sklearn.preprocessing import StandardScaler

# Load trained models
modeldir = r"D:\code\python\intership\main\emotion\models"
svm_model_path = os.path.join(modeldir, "mfcc_savee_trained-model.svm")

# Emotion Mapping
emotion_mapping = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Pleasant Surprised"
}

# Load Scaler
with open(os.path.join(modeldir, "scaler.pkl"), 'rb') as f:
    scaler = pickle.load(f)

# Extract features from audio bytes
def extract_features(audio_data, n_mfcc=13):
    y, sr = sf.read(io.BytesIO(audio_data))  # Read from byte stream
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # Convert stereo to mono if needed
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
    return mfccs.reshape(1, -1)

# Predict emotion using only the SVM model
def predict_emotion(audio_data):
    features = extract_features(audio_data)
    features = scaler.transform(features)

    try:
        with open(svm_model_path, 'rb') as file:
            svm_model = pickle.load(file)
        pred = svm_model.predict(features)[0]
        return emotion_mapping.get(int(pred), "Unknown")
    except Exception as e:
        print(f"❌ Error using SVM model: {e}")
        return "Error"
