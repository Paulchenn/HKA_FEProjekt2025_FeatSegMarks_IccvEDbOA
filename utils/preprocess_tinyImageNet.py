import os
import shutil

def prepare_train(train_dir):
    print(f"Preparing train directory: {train_dir}")
    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir)
        images_path = os.path.join(class_path, "images")
        if os.path.isdir(images_path):
            # Move all images up one level
            for img_file in os.listdir(images_path):
                shutil.move(os.path.join(images_path, img_file), class_path)
            os.rmdir(images_path)  # Remove the now-empty images folder

def prepare_val(val_dir):
    print(f"Preparing val directory: {val_dir}")
    images_dir = os.path.join(val_dir, "images")
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.exists(ann_file):
        print("val_annotations.txt not found, skipping val preparation.")
        return
    # Read annotations and move images into class folders
    with open(ann_file, "r") as f:
        for line in f:
            img, class_id, *_ = line.strip().split('\t')
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(images_dir, img)
            dst = os.path.join(class_dir, img)
            if os.path.exists(src):
                shutil.move(src, dst)
    # Optionally remove the images directory after moving
    if os.path.isdir(images_dir):
        os.rmdir(images_dir)

if __name__ == "__main__":
    # Change this path to your dataset location if needed
    cwd = os.getcwd()
    root = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(cwd)
            )
        ),
        "data"
    )
    print(root)
    prepare_train(os.path.join(root, "train"))
    prepare_val(os.path.join(root, "val"))
    print("Tiny ImageNet folder structure is now compatible with torchvision.datasets.ImageFolder.")