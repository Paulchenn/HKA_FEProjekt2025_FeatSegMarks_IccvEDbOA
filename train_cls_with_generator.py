import os
import time
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from utils.helpers import get_config, get_train_val_loaders, blur_image
from utils.modules import EMSE, DS_EMSE, TSD, TSG
from models import generation_imageNet_V2_2 as generation_imageNet


def train_classifier(config):

    # === Transforms for loading ImageNet ===
    transform_train = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # === Load dataset ===
    train_loader, val_loader = get_train_val_loaders(
        config,
        data_dir=config.DATASET_PATH,
        transform_train=transform_train,
        transform_val=transform_val,
        pin_memory=True
    )

    # === Generator & Preprocessing Modules ===
    ds_emse = DS_EMSE(config=config)
    tsd = TSD(config=config)
    tsg = TSG(config=config)

    netG = generation_imageNet.generator(img_size=config.IMG_SIZE,
                                         z_dim=config.NOISE_SIZE).to(config.DEVICE)
    if os.path.exists(config.PATH_TUNED_G):
        checkpoint = torch.load(config.PATH_TUNED_G, map_location=config.DEVICE)
        netG.load_state_dict(checkpoint, strict=False)
        print(f"[Init] Loaded Generator weights from {config.PATH_TUNED_G}")
    else:
        print(f"[Init] Not loaded any Generator weights from {config.PATH_TUNED_G}")
    netG.eval()

    # === Classifier ===
    cls_weights = ResNet18_Weights.DEFAULT
    cls = models.resnet18(weights=None)
    cls.fc = nn.Linear(cls.fc.in_features, config.NUM_CLASSES)
    cls = cls.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cls.parameters(), lr=config.LR_CLS, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_acc = 0.0

    for epoch in range(config.EPOCHS_CLS):
        cls.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", ncols=100)
        # i = 0
        for imgs, labels in pbar:
            # if i > 0:
            #     break
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)

            # === Pipeline: DS_EMSE → TSD → Generator ===
            with torch.no_grad():
                img_blur = blur_image(imgs, config.DOWN_SIZE)
                edgeMap = ds_emse.diff_edge_map(imgs)
                deformedEdge, tsd_grid = tsd.doTSD(edgeMap, return_grid=True)
                gen_imgs = tsg.generateImg(imgs.shape[0], netG, deformedEdge, img_blur)

            # === Forward through classifier ===
            outputs = cls(gen_imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             acc=f"{100.*correct/total:.2f}%")
            
            # i += 1

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # === Validation ===
        cls.eval()
        val_correct, val_total = 0, 0
        # i = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                # if i > 0:
                #     break
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)

                img_blur = blur_image(imgs, config.DOWN_SIZE)
                edgeMap = ds_emse.diff_edge_map(imgs)
                deformedEdge, tsd_grid = tsd.doTSD(edgeMap, return_grid=True)
                gen_imgs = tsg.generateImg(imgs.shape[0], netG, deformedEdge, img_blur)

                outputs = cls(gen_imgs)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

                # i += 1

        val_acc = 100. * val_correct / val_total
        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(cls.state_dict(), f"Result/cls_training/best_cls_{epoch}.pth")
            print(f"[Checkpoint] Saved new best model (acc={best_acc:.2f}%)")


if __name__ == "__main__":
    # === Load config ===
    from utils.helpers import get_config
    import json

    config = get_config("config/config.json")

    # Add classifier-specific params
    config.IMG_SIZE = getattr(config, "image_size", 256)
    config.LR_CLS = 1e-4
    config.EPOCHS_CLS = 200
    config.SAVE_PATH_CLS = getattr(config, "save_path", "./Result") + "/cls_training_imageNet"

    # Device
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Init] Training classifier on device {config.DEVICE}")
    train_classifier(config)
