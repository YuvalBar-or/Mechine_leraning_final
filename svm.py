import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_images_from_folders(main_folder_path, csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Ensure the 'path' and 'label' columns exist
    if 'path' not in df.columns or 'label' not in df.columns:
        raise ValueError("The CSV file must contain 'path' and 'label' columns.")

    # Create a dictionary to map image paths to their labels
    label_dict = dict(zip(df['path'], df['label']))

    images = []
    labels = []

    # Traverse the main folder and subfolders to load images
    for subdir, dirs, files in os.walk(main_folder_path):
        for file in files:
            # Construct the relative path used in the CSV
            relative_path = os.path.relpath(os.path.join(subdir, file), main_folder_path)
            if relative_path in label_dict:
                file_path = os.path.join(subdir, file)
                # Read and preprocess the image
                with Image.open(file_path) as img:
                    img = img.convert('L')  # Convert to grayscale
                    img = img.resize((64, 64))  # Resize to a fixed size
                    img = np.array(img).flatten()  # Flatten the image to a 1D array
                    images.append(img)
                    labels.append(label_dict[relative_path])

    return np.array(images), np.array(labels)


def apply_svm_and_print_accuracy(main_folder_path, csv_file_path):
    # Load images and labels
    X, y = load_images_from_folders(main_folder_path, csv_file_path)

    # Encode the labels to numeric values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Create an SVM classifier
    svm_classifier = SVC(kernel='linear')

    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = svm_classifier.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")


main_folder_path = 'dataset'
csv_file_path = 'data.csv'

apply_svm_and_print_accuracy(main_folder_path, csv_file_path)
