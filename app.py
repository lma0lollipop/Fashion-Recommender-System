import streamlit as st
from PIL import Image
import os

from model_utils import load_dataset_features, get_similar_images

st.set_page_config(
    page_title="Fashion Recommender",
    layout="wide"
)

st.title("👗 Fashion Image Recommender")
st.write("Upload a fashion image and get similar outfit recommendations.")

DATASET_PATH = "women fashion"

@st.cache_resource
def load_data():
    return load_dataset_features(DATASET_PATH)

all_features, all_paths = load_data()

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    query_image_path = "temp_query.jpg"
    with open(query_image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.subheader("Query Image")
    st.image(query_image_path, width=300)

    results = get_similar_images(
        query_image_path,
        all_features,
        all_paths,
        top_n=5
    )

    st.subheader("Similar Recommendations")

    cols = st.columns(5)
    for col, (img_path, score) in zip(cols, results):
        with col:
            st.image(Image.open(img_path), use_container_width=True)
            st.caption(f"Similarity: {score:.2f}")
