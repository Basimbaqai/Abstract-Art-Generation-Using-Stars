# Star Points Connection & Generation

This repository contains a Jupyter Notebook designed for generating, visualizing, and analyzing star point connections to form constellations. The project also uses Hugging Face's Diffusers library to generate artistic images based on constellation skeletons, providing a unique blend of computational geometry and AI-generated art.

## Features

- **Star Point Generation**: Create random or predefined star patterns.
- **Constellation Formation**: Connect points based on various algorithms or rules.
- **Visualization**: Interactive and static plots for clear representation.
- **Conditional Image Generation**: Use Hugging Face's `Stable Diffusion ControlNet` to generate abstract artistic representations of constellations based on their skeletons.
- **Customization**: Options to modify parameters like density, connectivity, and star properties.

## Models Used

This project leverages pre-trained models from the Hugging Face Diffusers library:
- **ControlNet Model**: [`lllyasviel/control_v11p_sd15_canny`](https://huggingface.co/lllyasviel/control_v11p_sd15_canny) for conditional image generation based on edge-detected images.
- **Stable Diffusion Pipeline**: [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) to create abstract artistic outputs guided by ControlNet.

## Installation

1. Clone this repository:
   ```bash
   git clone (https://github.com/Basimbaqai/Abstract-Art-Generation-Using-Stars.git)
   cd star-points-generation
2. Install the required Python packages
3. Ensure you have access to a GPU with CUDA support for optimal performance.

## Usage
- Step 1: Generate Star Skeletons
Use the notebook to generate star point connections and visualize the skeleton of constellations.
Save the skeleton image, e.g., connections_result.jpg.
- Step 2: Generate Abstract Art
Load the skeleton image and process it using Hugging Face's ControlNet model.
Use the provided example code snippet in the notebook 
