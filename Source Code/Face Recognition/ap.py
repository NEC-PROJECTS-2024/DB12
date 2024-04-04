from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

app = Flask(__name__)

# Function to preprocess an image
def preprocess_image(image_path, target_size=(100, 100)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = cv2.resize(image, target_size) / 255.0  # Normalize pixel values to [0, 1]
        image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale image
        return image
    else:
        return None

# Load trained CNN model
model_path = "trained_model/trained_model.h5"  # Replace with the path to your trained model
model = load_model(model_path)

# Function to generate embeddings for an image using the trained model
def generate_embedding(image):
    return model.predict(np.expand_dims(image, axis=0))

# Function to perform image matching
def image_matching(upload_image_path, dataset_dir, top_k=5):
    # Preprocess the uploaded image
    upload_image = preprocess_image(upload_image_path)
    if upload_image is None:
        return None

    # Generate embedding for the uploaded image
    upload_embedding = generate_embedding(upload_image)

    # Iterate through each image in the dataset directory
    similarity_scores = []
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            # Preprocess the dataset image
            dataset_image = preprocess_image(image_path)
            if dataset_image is None:
                continue
            # Generate embedding for the dataset image
            dataset_embedding = generate_embedding(dataset_image)
            # Calculate cosine similarity between the embeddings
            similarity = cosine_similarity(upload_embedding, dataset_embedding)[0][0]
            similarity_scores.append((person_name, filename, similarity))

    # Rank images based on similarity scores
    similarity_scores.sort(key=lambda x: x[2], reverse=True)

    # Return top k matches
    return similarity_scores[:top_k]

# Route for the home page
@app.route('/')
def home():
    return render_template('index1.html')

# Route for image upload and matching
@app.route('/match_images', methods=['POST'])
def match_images():
    # Get the uploaded image file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # Save the uploaded image
        uploaded_image_path = "uploaded_images/uploaded_image.jpg"  # Path to save the uploaded image
        uploaded_file.save(uploaded_image_path)

        # Perform image matching
        top_matches = image_matching(uploaded_image_path, "preprocessed_dataset")

        # Display top matched images
        image_paths = ["preprocessed_dataset/" + match[0] + "/" + match[1] for match in top_matches]
        titles = [f"Similarity: {match[2]:.4f}" for match in top_matches]

        return render_template('result.html', image_paths=image_paths, titles=titles)
    else:
        return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)
