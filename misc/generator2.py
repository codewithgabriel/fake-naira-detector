import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
from PIL import Image, ImageOps

# Set image dimensions
img_width = 224
img_height = 224

# Function to create mirrored folder structure
def create_fake_directory_structure(genuine_dir, fake_dir):
    for root, dirs, _ in os.walk(genuine_dir):
        for dir_name in dirs:
            genuine_subdir = os.path.join(root, dir_name)
            relative_path = os.path.relpath(genuine_subdir, genuine_dir)
            fake_subdir = os.path.join(fake_dir, relative_path)
            os.makedirs(fake_subdir, exist_ok=True)

# Function to apply fake transformations
def generate_fake_image(image):
    # Apply simple transformations to simulate fake images
    transformations = [
        lambda x: ImageOps.invert(x),  # Color inversion
        lambda x: x.rotate(45),        # Rotate by 45 degrees
        lambda x: ImageOps.mirror(x),  # Horizontal flip
        lambda x: ImageOps.posterize(x, bits=2)  # Posterization
    ]
    
    # Randomly select a transformation
    transform = np.random.choice(transformations)
    return transform(image)

# Function to generate fake notes
def generate_fake_notes(genuine_dir, fake_dir, num_fakes_per_image=1):
    create_fake_directory_structure(genuine_dir, fake_dir)
    
    for root, _, files in os.walk(genuine_dir):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                genuine_image_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(genuine_image_path, genuine_dir)
                fake_image_dir = os.path.dirname(os.path.join(fake_dir, relative_path))

                # Load the genuine image
                img = load_img(genuine_image_path, target_size=(img_height, img_width))
                img_array = img_to_array(img) / 255.0  # Normalize

                # Generate fake images
                for i in range(num_fakes_per_image):
                    fake_image = generate_fake_image(array_to_img(img_array))
                    
                    # Save fake image
                    fake_image_path = os.path.join(fake_image_dir, f"fake_{i}_{file_name}")
                    save_img(fake_image_path, fake_image)

# Paths
genuine_dir = 'genuine'
fake_dir = 'fake'

# Generate fake notes
generate_fake_notes(genuine_dir, fake_dir)
