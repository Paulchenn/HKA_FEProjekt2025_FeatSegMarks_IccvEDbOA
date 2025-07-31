import json
import os
import torch
import time
import itertools
import pdb

import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from types import SimpleNamespace
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from datetime import datetime
from PIL import Image


#def preperations():


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
    config["NUM_WORKERS"] = os.cpu_count()-1

    config["PATH_TUNED_D"] = config["path_tuned_D"]
    config["PATH_TUNED_G"] = config["path_tuned_G"]
    config["PATH_BEST_CLS"] = config["path_best_cls"]
    config["PATH_OPTIM_D"] = config["path_optim_D"]
    config["PATH_OPTIM_G"] = config["path_optim_G"]
    config["PATH_OPTIM_CLS"] = config["path_optim_CLS"]

    config["IMG_SIZE"] = config["image_size"]
    config["GEN_IN_DIM"] = config["generator_input_dim"]

    config["PATIENCE"] = config["patience"]
    config["acc_history"] = deque(maxlen=config["patience"])
    config["MIN_SLOPE"] = config["min_slope"]

    config["SHOW_IMG_EMSE_TSD"] = config["show_imgEmseTsd"]
    config["SHOW_TSG"] = config["show_tsg"]
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
    config["DATASET_PATH"] = config["dataset"]["path"]
    config["NUM_CLASSES"] = config["dataset"]["num_classes"]
    config["CLASS_NAMES"] = list(config["dataset"]["SELECTED_SYNSETS"].values())
    config["SELECTED_SYNSETS"] = config["dataset"]["SELECTED_SYNSETS"]

    today_str = datetime.today().strftime('%Y%m%d_%H%M%S')
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
    Returns train and validation DataLoaders from an ImageFolder dataset.
    Applies different transforms to train and val datasets.
    Optionally filters to only include selected classes.
    """
    # Full dataset (no transform yet)
    base_dataset = datasets.ImageFolder(data_dir)

    # Filter by selected class names (optional)
    class_names = set(config.CLASS_NAMES)
    if class_names is not None:
        # Map class name to index
        class_to_idx = base_dataset.class_to_idx
        selected_idx = [class_to_idx[name] for name in class_names if name in class_to_idx]

        # Filter samples
        filtered_samples = [s for s in base_dataset.samples if s[1] in selected_idx]

        # Create new dataset with filtered samples
        base_dataset.samples = filtered_samples
        base_dataset.targets = [s[1] for s in filtered_samples]

    # Split dataset
    total_size = len(base_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

    # Wrap subsets to apply transform
    train_subset.dataset.transform = transform_train
    val_subset.dataset.transform = transform_val

    # Create loaders
    print(f"Used num_workers: {config.NUM_WORKERS}")
    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False,
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


def show_imgEmseTsd(
    img,
    deformedImg,
    edgeMap
):
    """
    Show the original image, deformed image, and edge map.
    :param img: Original image tensor
    :param deformedImg: Deformed image tensor
    :param edgeMap: Edge map tensor
    """
    
    # === Show the original and deformed image ===
    plt.figure(figsize=(10, 5))  # Adjusted figure size
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('original image')
    plt.imshow(img[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('edge map')
    edgeMap_clipped = edgeMap[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    edgeMap_clipped = np.clip(edgeMap_clipped, 0, 1)
    plt.imshow(edgeMap_clipped)

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title('deformed image')
    deformedImg_clipped = deformedImg[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    deformedImg_clipped = np.clip(deformedImg_clipped, 0, 1)
    plt.imshow(deformedImg_clipped)

    plt.show(block=False)
    plt.pause(2)  # Show for 2 seconds (adjust as needed)
    plt.close()


def show_tsg(
    img,
    img_blur,
    img_for_loss,
    G_rough,
    G_fine,
    G_fine_resized,
    G_fine_norm,
    edge_map_from_syn
):
    """
    Show the original image, deformed image, and edge map.
    :param img:
    :param img_blur:
    :param img_for_loss:
    :param G_rough:
    :param G_fine:
    :param G_fine_resized:
    :param G_fine_norm:
    :param edge_map_from_syn
    """
    
    # === Show the original and deformed image ===
    plt.figure(figsize=(10, 5))  # Adjusted figure size
    plt.subplot(2, 4, 1)
    plt.axis('off')
    plt.title('original image')
    plt.imshow(np.clip(img[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0))

    plt.subplot(2, 4, 2)
    plt.axis('off')
    plt.title('blured image')
    plt.imshow(np.clip(img_blur[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0))

    plt.subplot(2, 4, 3)
    plt.axis('off')
    plt.title('image for loss')
    plt.imshow(np.clip(img_for_loss[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0))

    plt.subplot(2, 4, 4)
    plt.axis('off')
    plt.title('gen img s1')
    plt.imshow(np.clip(G_rough[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0))

    plt.subplot(2, 4, 5)
    plt.axis('off')
    plt.title('gen img s2')
    G_fine_f32 = G_fine.float()
    plt.imshow(np.clip(G_fine_f32[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0))

    plt.subplot(2, 4, 6)
    plt.axis('off')
    plt.title('resized img s2')
    plt.imshow(np.clip(G_fine_resized[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0))

    plt.subplot(2, 4, 7)
    plt.axis('off')
    plt.title('norm res img s2')
    plt.imshow(np.clip(G_fine_norm[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0))

    plt.subplot(2, 4, 8)
    plt.axis('off')
    plt.title('edge map gem img s2')
    plt.imshow(np.clip(edge_map_from_syn[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0))

    #pdb.set_trace()
    plt.show(block=False)
    plt.pause(2)  # Show for 2 seconds (adjust as needed)
    plt.close()


def show_result(
        config,
        num_epoch,
        img=None,
        edgeMap=None,
        deformedImg=None,
        path='Result/result.png',
        print_original=False,
        show=False,
        save=False,
        *,
        netG
):
    mn_batch = edgeMap.shape[0]

    if not print_original:
        zz = torch.randn(mn_batch, 100, 1, 1).to(config.DEVICE)
        netG.eval()
        test_images = netG(zz, edgeMap, deformedImg)
        netG.train()
        myTitle = 'Generated Images'
    else:
        test_images = img
        myTitle = 'Original Images'

    size_figure_grid = int(np.ceil(np.sqrt(mn_batch)))
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5), squeeze=False)
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(mn_batch):
        i = k // size_figure_grid
        j = k % size_figure_grid
        if i < size_figure_grid and j < size_figure_grid:
            ax[i, j].cla()
            img = test_images[k].cpu().data.numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img * 0.5) + 0.5
            img = np.clip(img, 0, 1)
            ax[i, j].imshow(img)

    label = 'Epoch {0}'.format(num_epoch+1)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if not path.endswith('.png'):
            path += '.png'
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.suptitle(myTitle, fontsize=16)
        print("Saving to:", os.path.abspath(path))
        plt.savefig(path, bbox_inches='tight', dpi=300)
        print(f"Result saved to {path}")

    if show:
        plt.show()
    else:
        plt.close()

