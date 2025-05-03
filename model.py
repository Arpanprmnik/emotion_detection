import os
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC  # SVM Classifier
from sklearn.neural_network import MLPClassifier  # MLP Classifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier  # Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors (KNN) Classifier

"""
Training Module and model saving
"""
def train_model(x_train, y_train, feature, emo_data_type):
    print(f"\n\nStart model building for feature {feature}")

    # SVM Start ==========
    C = 0.1  # SVM regularization parameter
    model_svm = SVC(kernel='linear', gamma=0.1, C=C)
    model_svm.fit(x_train, y_train)

    picklefile = os.path.join(modeldir, f"{feature}_{emo_data_type}_trained-model.svm")
    pickle.dump(model_svm, open(picklefile, 'wb'))
    print("SVM Model: Done")
    # SVM End ==========

    # MLP Start ==========
    model_mlp = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, 
                              hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    model_mlp.fit(x_train, y_train)

    picklefile = os.path.join(modeldir, f"{feature}_{emo_data_type}_trained-model.mlp")
    pickle.dump(model_mlp, open(picklefile, 'wb'))
    print("MLP Model: Done")
    # MLP End ==========

    # Logistic Regression Start =====
    model_reg = linear_model.LogisticRegression(solver='liblinear')
    model_reg.fit(x_train, y_train)

    picklefile = os.path.join(modeldir, f"{feature}_{emo_data_type}_trained-model.reg")
    pickle.dump(model_reg, open(picklefile, 'wb'))
    print("Logistic Regression Model: Done")
    # Logistic Regression End =====

    # Decision Tree Start =====
    model_dtree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    model_dtree.fit(x_train, y_train)

    picklefile = os.path.join(modeldir, f"{feature}_{emo_data_type}_trained-model.dt")
    pickle.dump(model_dtree, open(picklefile, 'wb'))
    print("Decision Tree Model: Done")
    # Decision Tree End =====

    # Random Forest Start =====
    model_rf = RandomForestClassifier(n_estimators=100)
    model_rf.fit(x_train, y_train)

    picklefile = os.path.join(modeldir, f"{feature}_{emo_data_type}_trained-model.rf")
    pickle.dump(model_rf, open(picklefile, 'wb'))
    print("Random Forest Model: Done")
    # Random Forest End =====

    # K-Nearest Neighbors (KNN) Start =====
    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(x_train, y_train)

    picklefile = os.path.join(modeldir, f"{feature}_{emo_data_type}_trained-model.knn")
    pickle.dump(model_knn, open(picklefile, 'wb'))
    print("KNN Model: Done")
    # K-Nearest Neighbors (KNN) End =====

def create_feat_csv_and_arrays(input_folder, output_file):
    """
    Reads .csv files, combines them into a single CSV and returns x_train, y_train arrays.
    """
    all_data = []

    # Iterate through all files in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            emotion_label = file_name.split('.')[0]
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
                            print(f"Warning: {file_name} has insufficient features. Skipped row.")
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")

    # Convert to NumPy arrays
    if all_data:
        all_data = np.array(all_data)
        x_train = all_data[:, :-1]  # Features
        y_train = all_data[:, -1]   # Labels

        # Convert to DataFrame and save to CSV
        column_names = [f"Feature_{i+1}" for i in range(13)] + ['Label']
        df = pd.DataFrame(all_data, columns=column_names)
        df.to_csv(output_file, index=False)
        print(f"Feature CSV saved to: {output_file}")
        return x_train, y_train
    else:
        print("No valid data found to save.")
        return np.array([]), np.array([])

# Define the emotion dictionary
emotion_dict = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'neutral': 4, 'Sad': 5, 'pleasant_surprised': 6
}

# Define the model directory
modeldir = r"D:\code\python\intership\main\frontend\emotion\models"

# Create the directory if it does not exist
if not os.path.exists(modeldir):
    os.makedirs(modeldir)
    print(f"Directory '{modeldir}' created successfully.")
else:
    print(f"Directory '{modeldir}' already exists.")

# Input folder and output file path
input_folder = r"D:\code\python\intership\main\frontend\emotion\train_fit"
output_file = r"D:\code\python\intership\main\frontend\emotion\train_fit.csv"

# Call the function to create CSV and get x_train and y_train
x_train, y_train = create_feat_csv_and_arrays(input_folder, output_file)

# Print the shapes of x_train and y_train
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Feature type and emotion data type
feature_type = "mfcc"
emo_data_type = "savee"

# Train the data and build models
train_model(x_train, y_train, feature_type, emo_data_type)
