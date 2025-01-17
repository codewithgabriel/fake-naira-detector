import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2

# Define source and target directories
genuine_folder = "genuine"  # Folder containing genuine notes
fake_folder = "fake"        # Target folder to save fake notes

# Ensure target folder structure exists
if not os.path.exists(fake_folder):
    os.makedirs(fake_folder)

# Loop through each subfolder in the genuine folder
for subfolder in os.listdir(genuine_folder):
    genuine_subfolder_path = os.path.join(genuine_folder, subfolder)
    fake_subfolder_path = os.path.join(fake_folder, subfolder)

    # Create corresponding fake subfolder
    if not os.path.exists(fake_subfolder_path):
        os.makedirs(fake_subfolder_path)

    # Process each image in the genuine subfolder
    for file_name in os.listdir(genuine_subfolder_path):
        image_path = os.path.join(genuine_subfolder_path, file_name)
        if not os.path.isfile(image_path):
            continue  # Skip non-files

        # Open the genuine image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1154, 556))  # Resize for uniform processing

        for i in range(5):  # Generate 5 fake images per genuine image
            # Clone the image for modifications
            modified_image = image.copy()

            # Apply random counterfeit techniques
            technique = np.random.choice(["blur", "noise", "brightness", "contrast", "distortion"])

            if technique == "blur":
                # Apply Gaussian blur
                modified_image = modified_image.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(1, 3)))

            elif technique == "noise":
                # Add random noise
                modified_image_np = np.array(modified_image)
                noise = np.random.normal(0, 25, modified_image_np.shape).astype('uint8')
                modified_image = Image.fromarray(np.clip(modified_image_np + noise, 0, 255))

            elif technique == "brightness":
                # Adjust brightness
                enhancer = ImageEnhance.Brightness(modified_image)
                modified_image = enhancer.enhance(np.random.uniform(0.5, 1.5))

            elif technique == "contrast":
                # Adjust contrast
                enhancer = ImageEnhance.Contrast(modified_image)
                modified_image = enhancer.enhance(np.random.uniform(0.5, 1.5))

            elif technique == "distortion":
                # Apply geometric distortion
                modified_image_np = np.array(modified_image)
                rows, cols, _ = modified_image_np.shape

                # Define source points as corners of the image
                src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])

                # Apply random shifts to create destination points
                dst_points = src_points + np.random.uniform(-10, 10, src_points.shape).astype(np.float32)

                # Ensure points are float32 and apply affine transform
                src_points = np.array(src_points, dtype=np.float32)
                dst_points = np.array(dst_points, dtype=np.float32)
                matrix = cv2.getAffineTransform(src_points, dst_points)
                modified_image_np = cv2.warpAffine(modified_image_np, matrix, (cols, rows))
                modified_image = Image.fromarray(modified_image_np)

            # Save the modified image
            fake_image_path = os.path.join(fake_subfolder_path, f"fake_{i}_{file_name}")
            modified_image.save(fake_image_path)

print("Fake images generated and saved.")
