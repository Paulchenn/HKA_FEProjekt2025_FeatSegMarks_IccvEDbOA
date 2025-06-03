import os
import tarfile
import requests
from tqdm import tqdm
import json
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the config directory
CONFIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'config')

# === Load parameters for ImageNet-Download from config_ImageNet.json ===
with open(os.path.join(CONFIG_DIR, 'config_ImageNet.json'), 'r') as f:
    config_ImageNet = json.load(f)
USERNAME = config_ImageNet["USERNAME"]
PASSWORD = config_ImageNet["PASSWORD"]
DATA_ROOT = config_ImageNet["DATA_ROOT"]
DATA_ROOT_FULL = os.path.join(os.path.dirname(SCRIPT_DIR), DATA_ROOT)
TRAIN_TAR_URL = config_ImageNet["TRAIN_TAR_URL"]
VAL_TAR_URL = config_ImageNet["VAL_TAR_URL"]
VAL_GT_FILE = config_ImageNet["VAL_GT_FILE"]
SELECTED_SYNSETS = config_ImageNet["SELECTED_SYNSETS"]

# === Download ImageNet from URL with Authentification ===
def download_file(url, output_path):
    if os.path.exists(output_path):
        print(f"{output_path} already downloaded.")
        return True
    with requests.get(url, stream=True, auth=(USERNAME, PASSWORD)) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), desc=f"Loading {os.path.basename(output_path)}"):
                f.write(chunk)

# === Extract training data ===
def extract_train_data():
    tar_path = os.path.join(DATA_ROOT_FULL)
    train_extract_dir = os.path.join(DATA_ROOT_FULL, "train")
    os.makedirs(train_extract_dir, exist_ok=True)
    
    if tar_path.endswith(".tar"):
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=DATA_ROOT_FULL)
    
    for synset in SELECTED_SYNSETS:
        class_tar = os.path.join(DATA_ROOT_FULL, 'train', f"{synset}")
        if not os.path.exists(class_tar):
            continue
        class_dir = os.path.join(train_extract_dir, synset)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir, exist_ok=True)
            with tarfile.open(class_tar) as t:
                t.extractall(path=class_dir)
        os.remove(class_tar)

# === Extract and sort validation data ===
def extract_val_data():
    tar_path = os.path.join(DATA_ROOT_FULL, "ILSVRC2012_img_val.tar")
    raw_dir = os.path.join(DATA_ROOT_FULL, "val_raw")
    sorted_dir = os.path.join(DATA_ROOT_FULL, "val")

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=raw_dir)

    # Load ground truth
    with open(VAL_GT_FILE, 'r') as f:
        gt_labels = [int(line.strip()) for line in f.readlines()]

    # Mapping from index to synset
    imagenet_map = {
        0: "n01440764",
        1: "n02123045",
        2: "n04379243",
        3: "n07942152",
        4: "n00021939"
    }

    files = sorted(os.listdir(raw_dir))
    os.makedirs(sorted_dir, exist_ok=True)
    for i, file in enumerate(files):
        label = gt_labels[i] - 1
        synset = imagenet_map.get(label)
        if synset in SELECTED_SYNSETS:
            dst_dir = os.path.join(sorted_dir, synset)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(os.path.join(raw_dir, file), os.path.join(dst_dir, file))
    shutil.rmtree(raw_dir)

# === Prepare dataloader ===
def get_loader(
    subdir,
    myTransform,
    myBatchSize=64,
    myShuffle=False,
):
    path = os.path.join(DATA_ROOT_FULL, subdir)

    dataset = datasets.ImageFolder(path, transform=myTransform)

    return DataLoader(dataset, batch_size=myBatchSize, shuffle=myShuffle)

# === Main-function ===
def main(
    transform_train,
    transform_test,
    batch_size=64
):
    os.makedirs(DATA_ROOT_FULL, exist_ok=True)

    print("==> Loading training data")
    alreadyLoaded_train = download_file(TRAIN_TAR_URL, DATA_ROOT_FULL)
    if alreadyLoaded_train == False:
        extract_train_data()

    print("==> Loading validation data")
    alreadyLoaded_test = download_file(VAL_TAR_URL, DATA_ROOT_FULL)
    if alreadyLoaded_test == False:
        extract_val_data()

    print("==> Initializing dataloaders")
    train_loader = get_loader(
        "train",
        myTransform=transform_train,
        myBatchSize=batch_size,
        myShuffle=True
    )
    val_loader = get_loader(
        "val",
        myTransform=transform_test,
        myBatchSize=batch_size,
        myShuffle=False
    )

    print(f"Train: {len(train_loader.dataset)} Pictures")
    print(f"Val: {len(val_loader.dataset)} Pictures")

    # Beispielnutzung (GAN-Training o.ä.)
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = main()
