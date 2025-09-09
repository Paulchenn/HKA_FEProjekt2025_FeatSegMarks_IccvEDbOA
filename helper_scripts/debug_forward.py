from utils.helpers import *
from utils.modules import *
from models import generation_imageNet_V1 as generation_imageNet
from types import SimpleNamespace
import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt

def show_tensor_image(img: torch.Tensor, title: str = "Original"):
    """
    Zeigt ein Tensor-Bild [1,3,H,W] oder [3,H,W] als RGB an.
    Handhabt ggf. [-1,1]-Normalisierung automatisch.
    """
    if img.dim() == 4:
        img = img[0]  # [3,H,W]
    img = img.detach().cpu()
    # Falls noch in [-1,1]: auf [0,1] zurückskalieren
    if img.min() < 0:
        img = (img + 1) / 2
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()  # -> [H,W,3]

    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

def load_img_from_config(config) -> torch.Tensor:
    """
    Lädt das Bild aus config.PICTURE_PATH, skaliert auf config.IMG_SIZE
    und gibt einen Tensor [1,3,H,W] (Werte in [0,1]) auf config.DEVICE zurück.
    """
    path = os.path.expanduser(config.PICTURE_PATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Bild nicht gefunden: {path}")

    pil_img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # -> [0,1]
        # Falls dein Netz [-1,1] erwartet, entkommentiere die nächste Zeile:
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = tfm(pil_img).unsqueeze(0)  # [1,3,H,W]
    return img.to(config.DEVICE)


config = SimpleNamespace(
    DEVICE='cpu',
    IMG_SIZE=256,
    DOWN_SIZE=128,
    BATCH_SIZE=16,
    PICTURE_PATH="~/torch/data/ImageNet256/abacus/014.jpg",
)
path2checkpoint = "./Result/20250818_182611/ImageNet256/tuned_netG/netG__stage1__Epoch_013.pth"

netG    = generation_imageNet.generator(d=128, img_size=256)

checkpoint = torch.load(path2checkpoint, map_location=config.DEVICE)
model_dict = netG.state_dict()
filtered_dict = {
    k: v for k, v in checkpoint.items()
    if k in model_dict and v.shape == model_dict[k].shape
}
skipped = [k for k in checkpoint if k not in filtered_dict]
if skipped:
    print(f"Not loaded Layer (due to conflicts in name and dimension):")
    for k in skipped:
        print(f"  - {k}  (Checkpoint shape: {checkpoint[k].shape})")
model_dict.update(filtered_dict)
netG.load_state_dict(model_dict)
print(f"Loaded Generator-Net checkpoint from {path2checkpoint}")

IMG = load_img_from_config(config)
print("IMG shape:", tuple(IMG.shape), "device:", IMG.device)
show_tensor_image(IMG, title="Originalbild")


emse = EMSE(config=config)
EDGE_MAP_TEST = emse.doEMSE(IMG)
show_tensor_image(EDGE_MAP_TEST, title="EMSE")

show_result(  # Shows the result of actual epoch
    config=config,
    num_epoch=-1,
    img=IMG,
    edgeMap=EDGE_MAP_TEST,
    netG=netG,
    show=True,
    save=False,
    path=None
)