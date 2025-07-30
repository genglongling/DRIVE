import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

# Directory containing the frames
frames_dir = "frames"

# Load all frame file paths
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])

# Load the images and convert them to NumPy arrays
images = [np.array(Image.open(frame)) for frame in frame_files]

# Define grid size for plotting (e.g., based on the number of frames)
num_frames = len(images)
grid_cols = math.ceil(math.sqrt(num_frames))  # Columns
grid_rows = math.ceil(num_frames / grid_cols)  # Rows

# Create the plot
fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 15))
axes = axes.flatten()  # Flatten to iterate easily

# Plot each frame in its respective grid cell
for i, img in enumerate(images):
    axes[i].imshow(img)
    axes[i].axis('off')  # Hide axes
    axes[i].set_title(f"Frame {i + 1}")

# Hide any unused subplot axes
for i in range(len(images), len(axes)):
    axes[i].axis('off')

# Adjust layout
plt.tight_layout()

# Save the combined image
output_path = "./visualization/demo.png"
plt.savefig(output_path, dpi=300)
plt.close()  # Close the plot to release memory

print(f"Visualization saved as {output_path}")
