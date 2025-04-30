import os
import pickle
import numpy as np
import pandas as pd
import librosa
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define the emotion dictionary (all lowercase for consistency)
emotion_dict = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'neutral': 4, 'sad': 5, 'pleasant_surprised': 6
}

# Directory to save models
modeldir = r"D:\code\python\intership\main\emotion\models"

if not os.path.exists(modeldir):
    os.makedirs(modeldir)
    print(f"Created model directory: {modeldir}")

# Feature Extraction Function (Enhanced)
def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)

    features = np.hstack([mfccs, chroma, contrast, mel])  # Combine all features
    return features

# Load CSV Data
def load_training_data(input_folder):
    all_data = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            emotion_label = file_name.split('.')[0].lower()  # Convert to lowercase
            if emotion_label in emotion_dict:
                y_value = emotion_dict[emotion_label]
                file_path = os.path.join(input_folder, file_name)

                try:
                    data = np.loadtxt(file_path, delimiter=',')
                    for row in data:
                        if len(row) >= 13:
                            features = row[:13]
                            features = np.append(features, y_value)
                            all_data.append(features)
                        else:
                            print(f"Warning: {file_name} has insufficient features.")

                except Exception as e:
                    print(f"Error reading {file_name}: {e}")

    if all_data:
        all_data = np.array(all_data)
        x_train = all_data[:, :-1]  # Features
        y_train = all_data[:, -1]   # Labels
        return x_train, y_train
    else:
        return np.array([]), np.array([])

# Train & Save Models
def train_and_save_models(x_train, y_train):
    print("\nStarting Model Training...\n")

    # Standardize Data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # Save Scaler
    with open(os.path.join(modeldir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # Train models
    models = {
        "SVM": SVC(kernel='linear', gamma=0.1, C=0.1),
        "MLP": MLPClassifier(alpha=0.01, batch_size=256, hidden_layer_sizes=(300,), max_iter=500),
        "Logistic Regression": linear_model.LogisticRegression(solver='liblinear'),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=3),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=3)
    }

    for name, model in models.items():
        model.fit(x_train, y_train)
        model_path = os.path.join(modeldir, f"mfcc_savee_trained-model.{name.lower()}")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"{name} Model Saved at {model_path}.")

# Load Data and Train
input_folder = r"D:\code\python\intership\main\emotion\train_fit"
x_train, y_train = load_training_data(input_folder)

if x_train.size > 0:
    train_and_save_models(x_train, y_train)
else:
    print("No valid training data found.")
