import streamlit as st
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image


# Function to process the image
def process_image(image_path, threshold_value, n_neighbors):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to find contours
    _, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract contour centers
    contour_centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            contour_centers.append((cx, cy))

    # Convert contour centers to numpy array
    contour_centers = np.array(contour_centers)

    # Perform K-Nearest Neighbors to find connections
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(contour_centers)
    connections = knn.kneighbors(contour_centers, return_distance=False)

    # Create a black image to visualize connections
    height, width = image.shape
    black_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw connections
    for i, neighbors in enumerate(connections):
        point1 = contour_centers[i]
        for neighbor_index in neighbors[1:]:  # Skip the first neighbor (itself)
            point2 = contour_centers[neighbor_index]
            cv2.line(black_image, tuple(point1), tuple(point2), (0, 0, 0), 10)

    return image, black_image


# Streamlit UI
def main():
    st.title("Connecting Star points Using Nearest Neighbours")

    # File uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load the uploaded image
        image_path = uploaded_file.name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Threshold slider
        threshold_value = st.slider("Threshold Value", min_value=0, max_value=255, value=230, step=1)

        # Number of neighbors slider
        n_neighbors = st.slider("Number of Neighbors", min_value=2, max_value=10, value=3, step=1)

        # Process the image
        original_image, processed_image = process_image(image_path, threshold_value, n_neighbors)

        # Display the images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image", use_container_width=True, channels="GRAY")
        with col2:
            st.image(processed_image, caption="Processed Image", use_container_width=True)


if __name__ == "__main__":
    main()
