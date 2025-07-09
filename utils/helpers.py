import json
import os
import torch
import pdb

import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from types import SimpleNamespace
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from datetime import datetime


def get_config(file_path):
    """
    Reads a configuration file and returns the configuration as a dictionary.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        SimpleNamespace: Configuration parameters as attributes.
    """
    with open(file_path, 'r') as f:
        config = json.load(f)

    config["DEVICE"] = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    config["BATCH_SIZE"] = config["batch_size"]
    config["EPOCHS"] = config["epochs"]
    config["NUM_WORKERS"] = config["num_workers"]

    config["PATH_TUNED_D"] = config["path_tuned_D"]
    config["PATH_TUNED_G"] = config["path_tuned_G"]
    config["PATH_OPTIM_D"] = config["path_optim_D"]
    config["PATH_OPTIM_G"] = config["path_optim_G"]

    config["IMG_SIZE"] = config["image_size"]
    config["GEN_IN_DIM"] = config["generator_input_dim"]

    config["PATIENCE"] = config["patience"]
    config["acc_history"] = deque(maxlen=config["patience"])
    config["MIN_SLOPE"] = config["min_slope"]

    config["SHOW_IMAGES"] = config["show_images"]
    config["SHOW_IMAGES_INTERVAL"] = config["show_images_interval"]

    config["LOG_INTERVAL"] = config["log_interval"]

    config["TRAIN_CLS"] = config["train_cls"]

    config["DEBUG_MODE"] = config["debug_mode"]
    config["DEBUG_ITERS_START"] = config["debugIterations_strt"]
    config["DEBUG_ITERS_AMOUNT"] = config["debugIterations_amount"]
    config["LR_GEN"] = config["learning_rate"]["generator"]
    config["LR_DISC"] = config["learning_rate"]["discriminator"]
    config["LR_CLS"] = config["learning_rate"]["classifier"]

    config["DATASET_NAME"] = config["dataset"]["name"]
    config["NUM_CLASSES"] = config["dataset"]["num_classes"]
    config["CLASS_NAMES"] = list(config["dataset"]["SELECTED_SYNSETS"].values())
    config["SELECTED_SYNSETS"] = config["dataset"]["SELECTED_SYNSETS"]

    today_str = datetime.today().strftime('%Y_%m_%d')
    config["SAVE_PATH"] = os.path.join(config["save_path"], today_str)

    return SimpleNamespace(**config)


def get_tiny_imagenet_loaders(
    root_dir,
    img_size=64,
    batch_size=64,
    num_workers=4,
    pin_memory=True
):
    """
    Returns train and validation DataLoaders for Tiny ImageNet.
    Assumes train and val folders are in ImageFolder format.
    """
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")

    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print("Number of train classes:", len(train_dataset.classes))
    print("Number of train images:", len(train_dataset.samples))
    print("First 5 train samples:", train_dataset.samples[:5])

    print("Number of val classes:", len(val_dataset.classes))
    print("Number of val images:", len(val_dataset.samples))
    print("First 5 val samples:", val_dataset.samples[:5])

    return train_loader, val_loader


def get_train_val_loaders(
        config,
        data_dir,
        transform_train,
        transform_val,
        val_split=0.2,
        pin_memory=True,
        seed=42
    ):
    """
    Splits the dataset in data_dir into train and validation sets and returns their DataLoaders.
    Applies transform_train to train set and transform_val to val set.
    """

    selected_classes = set(config.CLASS_NAMES)

    # Load the full dataset with a dummy transform (will be replaced per split)
    # Load the full dataset with no transform (we'll apply transforms per split)
    full_dataset = datasets.ImageFolder(data_dir, transform=None)

    # Map class name to index in ImageFolder
    class_to_idx = {v: k for k, v in full_dataset.class_to_idx.items()}
    selected_indices = [idx for idx, classname in class_to_idx.items() if classname in selected_classes]
    #selected_indices = [class_to_idx[name] for name in selected_classes if name in class_to_idx]

    # Filter images to only include selected class indices
    filtered_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in selected_indices]
    filtered_dataset = Subset(full_dataset, filtered_indices)

    # Now split filtered dataset
    total_size = len(filtered_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    train_subset, val_subset = random_split(
        filtered_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Subset class that allows individual transforms
    class SubsetWithTransform(Subset):
        def __init__(self, subset, transform):
            super().__init__(subset.dataset, subset.indices)
            self.transform = transform
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        
    train_dataset = SubsetWithTransform(train_subset, transform_train)
    val_dataset = SubsetWithTransform(val_subset, transform_val)

    train_dataset.dataset.classes = list(selected_classes)
    val_dataset.dataset.classes = list(selected_classes)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=pin_memory)
    
    # After creating the loaders
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")

    # Check a batch
    img_train, label_train = next(iter(train_loader))
    print("Train batch shape:", img_train.shape, "Labels:", label_train)

    img_val, label_val = next(iter(val_loader))
    print("Val batch shape:", img_val.shape, "Labels:", label_val)

    #pdb.set_trace()  # Start debugging
    print("Number of images in train_loader:", len(train_loader.dataset))
    print("Number of classes in train_loader:", len(train_loader.dataset.dataset.classes))
    print("Number of images in val_loader:", len(val_loader.dataset))
    print("Number of classes in val_loader:", len(val_loader.dataset.dataset.classes))

    if False:
        # Check class names
        if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset, 'classes'):
            print("Classes:", train_loader.dataset.dataset.classes)

        # Select the first image in the train batch
        img_train_show = img_train[0].cpu().numpy()  # shape: (C, H, W)
        img_train_show = img_train_show * 0.5 + 0.5  # Undo normalization
        img_train_show = np.transpose(img_train_show, (1, 2, 0))  # (H, W, C)

        # Select the first image in the val batch
        img_val_show = img_val[0].cpu().numpy()
        img_val_show = img_val_show * 0.5 + 0.5
        img_val_show = np.transpose(img_val_show, (1, 2, 0))

        # Show both images in a subplot
        _, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img_train_show)
        axs[0].set_title(f"Train Label: {label_train[0].item()}")
        axs[0].axis('off')
        axs[1].imshow(img_val_show)
        axs[1].set_title(f"Val Label: {label_val[0].item()}")
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

    return train_loader, val_loader