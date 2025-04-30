import pickle
import os
import pandas as pd
from sklearn.metrics import accuracy_score

# Path to the directory where models are saved
modeldir = r"D:\code\python\intership\main\emotion\models"

# Define model file paths
models = {
    "SVM": os.path.join(modeldir, "mfcc_savee_trained-model.svm"),
    "MLP": os.path.join(modeldir, "mfcc_savee_trained-model.mlp"),
    "Logistic Regression": os.path.join(modeldir, "mfcc_savee_trained-model.reg"),
    "Decision Tree": os.path.join(modeldir, "mfcc_savee_trained-model.dt"),
    "Random Forest": os.path.join(modeldir, "mfcc_savee_trained-model.rf"),
    "KNN": os.path.join(modeldir, "mfcc_savee_trained-model.knn")
}


# Path to the CSV file containing the test dataset
test_csv_path = r"D:\code\python\intership\main\emotion\evolution_fit.csv"

# Load the test dataset, skipping the first row (header)
data = pd.read_csv(test_csv_path, skiprows=1, header=None)

# Use the first 13 columns as features (X) and the last column as labels (y)
X_test = data.iloc[:, :-1].values  # First 13 columns as features
y_test = data.iloc[:, -1].values   # Last column as labels (labels)

# Verify the dataset structure
# print("Dataset Preview:")
# print(data.head())

# Check feature and label consistency
if len(X_test) != len(y_test):
    print(f"Error: Mismatch in samples between X_test ({len(X_test)}) and y_test ({len(y_test)})")
else:
    # Iterate through models and calculate predictions
    for model_name, model_path in models.items():
        try:
            # Load the model
            with open(model_path, 'rb') as file:
                model = pickle.load(file)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred) * 100
            print(f"{model_name} Accuracy: {accuracy:.2f}%")

        except Exception as e:
            print(f"Error testing {model_name}: {e}")