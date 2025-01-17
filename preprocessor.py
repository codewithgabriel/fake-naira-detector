import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import random

# Paths
genuine_folder = "genuine"
counterfeit_folder = "counterfeit"

# Function to create the folder structure for counterfeit
def create_counterfeit_folders(genuine_path, counterfeit_path):
    if not os.path.exists(counterfeit_path):
        os.makedirs(counterfeit_path)
    for subdir in os.listdir(genuine_path):
        subdir_path = os.path.join(genuine_path, subdir)
        if os.path.isdir(subdir_path):
            os.makedirs(os.path.join(counterfeit_path, subdir), exist_ok=True)

# Function to apply varying "fake" filters
def apply_varied_filter(input_image_path, output_image_path):
    # Open the image
    img = Image.open(input_image_path)

    # Convert to grayscale
    img = img.convert("L")

    # Random sepia/colormap effect
    sepia = np.array(img)
    colormap_choice = random.choice([cv2.COLORMAP_OCEAN, cv2.COLORMAP_BONE, cv2.COLORMAP_WINTER, cv2.COLORMAP_SUMMER])
    sepia = cv2.applyColorMap(sepia, colormap_choice)

    # Convert back to PIL Image
    sepia_img = Image.fromarray(sepia)

    # Randomly adjust contrast
    contrast_factor = random.uniform(0.5, 1.2)
    contrast_enhancer = ImageEnhance.Contrast(sepia_img)
    sepia_img = contrast_enhancer.enhance(contrast_factor)

    # Randomly adjust brightness
    brightness_factor = random.uniform(0.8, 1.5)
    brightness_enhancer = ImageEnhance.Brightness(sepia_img)
    sepia_img = brightness_enhancer.enhance(brightness_factor)

    # Add random noise
    sepia_np = np.array(sepia_img)
    noise_intensity = random.randint(10, 40)  # Random noise intensity
    noise = np.random.normal(0, noise_intensity, sepia_np.shape).astype(np.uint8)
    sepia_np = cv2.addWeighted(sepia_np, 0.9, noise, 0.1, 0)

    # Add random blurring to simulate wear and tear
    if random.random() > 0.5:  # Randomly decide to blur
        blur_intensity = random.choice([3, 5, 7])  # Random blur kernel size
        sepia_np = cv2.GaussianBlur(sepia_np, (blur_intensity, blur_intensity), 0)

    # Save the final image
    final_image = Image.fromarray(sepia_np)
    final_image.save(output_image_path)

# Main processing function
def process_images(genuine_path, counterfeit_path):
    create_counterfeit_folders(genuine_path, counterfeit_path)
    for subdir, _, files in os.walk(genuine_path):
        for file in files:
            input_image_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(subdir, genuine_path)
            output_subdir = os.path.join(counterfeit_path, relative_path)
            os.makedirs(output_subdir, exist_ok=True)
            output_image_path = os.path.join(output_subdir, file)

            print(f"Processing: {input_image_path} -> {output_image_path}")
            apply_varied_filter(input_image_path, output_image_path)

# Run the script
genuine_path = "genuine"
counterfeit_path = "counterfeit"
process_images(genuine_path, counterfeit_path)
