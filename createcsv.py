import soundfile
import os
import librosa
import numpy as np
import pandas as pd
import glob

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name):
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T
            return mfccs
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None

def process_folder(input_folder, output_folder):
    try:
        # Get all subfolders in the input folder
        subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

        for subfolder in subfolders:
            # Get the name of the emotion (folder name)
            emotion = os.path.basename(subfolder)

            print(f"Processing folder: {emotion}")

            # Find all .wav files in the current subfolder
            files_grabbed = [f for f in glob.glob(os.path.join(subfolder, "*.wav"), recursive=True)]

            if not files_grabbed:
                print(f"No .wav files found in directory: {subfolder}")
                continue

            # Extract features from each file
            features = []
            for file in files_grabbed:
                feature = extract_feature(file)
                if feature is not None:
                    features.append(feature)

            if features:
                # Save the features to a CSV file named after the emotion
                output_file = os.path.join(output_folder, f"{emotion}.csv")
                df = pd.DataFrame(np.concatenate(features, axis=0))
                df.to_csv(output_file, index=False, header=False)
                print(f"Features for {emotion} saved to {output_file}")
            else:
                print(f"No features extracted for {emotion}. Skipping.")

    except Exception as e:
        print(f"Error Message: {str(e)}")

if __name__ == '__main__':
    input_folder = r"D:\code\python\intership\main\frontend\emotion\evolution"  # Base folder containing subfolders for each emotion
    output_folder = r"D:\code\python\intership\main\frontend\emotion\evolution_fit"  # Directory to save the output CSV files

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_folder(input_folder, output_folder)