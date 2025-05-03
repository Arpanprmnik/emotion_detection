import soundfile
import os
import librosa
import numpy as np
import pandas as pd
import glob


def create_feat_csv_and_arrays(input_folder, output_file):
    """
    Reads .csv files, combines them into a single CSV, and returns x_train, y_train arrays.

    Args:
        input_folder (str): Path to the folder containing .csv files.
        output_file (str): Path to the output CSV file.

    Returns:
        x_train (np.ndarray): Array containing the first 13 features.
        y_train (np.ndarray): Array containing the corresponding labels.
    """
    all_data = []  # To hold combined data

    # Iterate through all files in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):  # Process only .csv files
            # Extract emotion label from the file name
            emotion_label = file_name.split('.')[0]  # e.g., angry.csv -> angry
            if emotion_label in emotion_dict:  # Ensure valid emotion
                y_value = emotion_dict[emotion_label]

                # Read the file content
                file_path = os.path.join(input_folder, file_name)
                try:
                    # Load as numpy array; specify comma as the delimiter
                    data = np.loadtxt(file_path, delimiter=',')

                    # Check if data has 13 or more features
                    for row in data:
                        if len(row) >= 13:
                            # Extract first 13 features and append label as y_train
                            features = row[:13]  # Take first 13 elements
                            features = np.append(features, y_value)  # Add label
                            all_data.append(features)
                        else:
                            print(f"Warning: {file_name} has insufficient features. Skipped row.")
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")

    # Convert to NumPy arrays
    if all_data:
        all_data = np.array(all_data)
        x_train = all_data[:, :-1]  # First 13 columns as features
        y_train = all_data[:, -1]   # Last column as labels

        # Convert to DataFrame to save to CSV
        column_names = [f"Feature_{i+1}" for i in range(13)] + ['Label']
        df = pd.DataFrame(all_data, columns=column_names)

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Feature CSV saved to: {output_file}")
        return x_train, y_train
    else:
        print("No valid data found to save.")
        return np.array([]), np.array([])

# Define the emotion dictionary
emotion_dict = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
                'neutral': 4, 'Sad': 5, 'pleasant_surprised': 6}

# Input folder and output file path
input_folder = r"D:\code\python\intership\main\frontend\emotion\evolution_fit"  # Replace with your folder path
output_file = r"D:\code\python\intership\main\frontend\emotion\evolution_fit.csv"

# Call the function to generate train_fit.csv
x_train, y_train = create_feat_csv_and_arrays(input_folder, output_file)

# Print the shapes of x_train and y_train for confirmation
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")