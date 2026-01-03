import os
import numpy as np
from PIL import Image

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# Load model once
model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
model.trainable = False


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    return img_array


def extract_features(image_path):
    img_array = preprocess_image(image_path)
    features = model.predict(img_array, verbose=0)
    return features.flatten()


def load_dataset_features(dataset_path):
    image_files = sorted([
        f for f in os.listdir(dataset_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    features = []
    paths = []

    for img in image_files:
        path = os.path.join(dataset_path, img)
        try:
            feat = extract_features(path)
            features.append(feat)
            paths.append(path)
        except:
            pass

    return np.array(features), paths


def get_similar_images(query_image, all_features, all_paths, top_n=5):
    query_feat = extract_features(query_image).reshape(1, -1)
    scores = cosine_similarity(query_feat, all_features)[0]
    indices = scores.argsort()[-top_n-1:-1][::-1]

    return [(all_paths[i], scores[i]) for i in indices]
