import os
import shutil

# Path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the config directory
CONFIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'config')

# Define paths
val_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'src/tinyImageNet/val')
images_dir = os.path.join(val_dir, 'images')
annotations_file = os.path.join(val_dir, 'val_annotations.txt')

# Target directory for sorted validation images
sorted_dir = os.path.join(val_dir, 'sorted_images')
os.makedirs(sorted_dir, exist_ok=True)

# Read annotations and move images to class-specific folders
with open(annotations_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        image_name, class_id = parts[0], parts[1]
        class_dir = os.path.join(sorted_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)
        src_path = os.path.join(images_dir, image_name)
        dst_path = os.path.join(class_dir, image_name)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
