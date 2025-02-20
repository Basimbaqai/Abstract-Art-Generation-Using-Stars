import streamlit as st
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


def process_image(image_path, threshold_value, n_neighbors):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    resize_width, resize_height = 350, 350
    image = cv2.resize(image, (resize_width, resize_height))

    _, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract contour centers
    contour_centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            contour_centers.append((cx, cy))

    contour_centers = np.array(contour_centers)

    if len(contour_centers) == 0:
        raise ValueError("No valid contours detected in the image.")

    # Check if there are enough points for the specified number of neighbors
    if len(contour_centers) < n_neighbors:
        raise ValueError(
            f"Not enough points ({len(contour_centers)}) for the specified number of neighbors ({n_neighbors})."
        )

    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(contour_centers)
    connections = knn.kneighbors(contour_centers, return_distance=False)

    height, width = image.shape
    black_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    for i, neighbors in enumerate(connections):
        point1 = contour_centers[i]
        for neighbor_index in neighbors[1:]:  # Skip the first neighbor (itself)
            point2 = contour_centers[neighbor_index]
            cv2.line(black_image, tuple(point1), tuple(point2), (0, 0, 0), 10)

    return image, black_image


# Streamlit UI
def main():
    st.title("Connecting Star Points Using Nearest Neighbors")

    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:

        image_path = uploaded_file.name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        threshold_value = st.slider("Threshold Value", min_value=0, max_value=255, value=230, step=1)

        n_neighbors = st.slider("Number of Neighbors", min_value=2, max_value=10, value=3, step=1)

        try:

            original_image, processed_image = process_image(image_path, threshold_value, n_neighbors)

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original Image (Resized to 512x512)", use_container_width=True,
                         channels="GRAY")
            with col2:
                st.image(processed_image, caption="Processed Image", use_container_width=True)

        except ValueError as e:

            st.error(str(e))


if __name__ == "__main__":
    main()
