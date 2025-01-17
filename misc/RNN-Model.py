import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess images
def load_images(directory, img_size=(128, 128)):
    data = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                data.append(img)
            else:
                print(f"Could not read image: {filepath}")
    print(f"Total images loaded: {len(data)}")
    return np.array(data)

# Feature extraction using HOG
def extract_hog_features(images):
    if len(images) == 0:
        print("No images available for feature extraction.")
        return np.array([])
    
    hog = cv2.HOGDescriptor()
    features = [hog.compute(img).flatten() for img in images]
    print(f"Extracted HOG features with shape: {len(features)}")
    return np.array(features)

# Define the RNN Autoencoder
def build_rnn_autoencoder(input_dim, timesteps):
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(timesteps, input_dim), return_sequences=True),
        LSTM(64, activation='relu', return_sequences=False),
        RepeatVector(timesteps),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(128, activation='relu', return_sequences=True),
        TimeDistributed(Dense(input_dim)),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Main script
if __name__ == "__main__":
    # Paths
    data_dir = "genuine/"  # Update this to the directory containing genuine notes

    # Step 1: Load and preprocess images
    images = load_images(data_dir)
    if images.size == 0:
        raise ValueError("Image loading failed. Check your directory structure or file permissions.")

    # Step 2: Extract features
    features = extract_hog_features(images)
    if features.size == 0:
        raise ValueError("Feature extraction failed. Ensure the images are valid.")

    # Step 3: Normalize features
    scaler = StandardScaler()
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)
    features_normalized = scaler.fit_transform(features)

    # Step 4: Reshape for RNN input
    timesteps = 10  # Adjust based on dataset
    samples = features_normalized.shape[0] // timesteps
    if samples == 0:
        raise ValueError("Insufficient data for RNN. Ensure enough images are provided.")
    rnn_input = features_normalized[:samples * timesteps].reshape(samples, timesteps, -1)

    # Step 5: Train/Test Split
    X_train, X_test = train_test_split(rnn_input, test_size=0.2, random_state=42)

    # Step 6: Build and train RNN Autoencoder
    input_dim = rnn_input.shape[2]
    model = build_rnn_autoencoder(input_dim, timesteps)
    model.summary()
    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

    # Step 7: Evaluate model
    reconstruction = model.predict(X_test)
    reconstruction_loss = np.mean(np.square(X_test - reconstruction), axis=(1, 2))
    threshold = np.percentile(reconstruction_loss, 95)  # Choose a suitable threshold
    print(f"Threshold for anomaly detection: {threshold}")

    # Step 8: Test on new images
    # Load new notes, preprocess, and compute reconstruction loss to detect anomalies.
    print("Pipeline complete. Use the trained model to test new notes.")

import matplotlib.pyplot as plt

plt.hist(reconstruction_loss, bins=30, alpha=0.7, color='blue')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("Reconstruction Loss Distribution")
plt.xlabel("Reconstruction Loss")
plt.ylabel("Frequency")
plt.legend()
plt.show()