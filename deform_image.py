# deform_image.py

import torch
from torchvision import transforms
from PIL import Image

# Importiere deine Generator-Klasse
from models.generation import generator

# Importiere die Funktionen für Edge und Info
from train_SDbOA_cifar import get_edge, get_info, TPS_Batch

def load_generator(model_path: str, device: torch.device) -> torch.nn.Module:
    netG = generator().to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    return netG

def deform_image(
    netG: torch.nn.Module,
    img: Image.Image,
    device: torch.device,
    sigma: float = 1.0,
    high_threshold: float = 0.3,
    low_threshold: float = 0.2,
    re12_transform=None,
    re32_transform=None
) -> Image.Image:
    # a) PIL → Tensor (normiert auf [0,1])
    to_tensor = transforms.ToTensor()
    img_t = to_tensor(img).unsqueeze(0).to(device)  # Form: (1, 3, H, W)

    # b) Edge-Map
    edge_map = get_edge(
        img_t,
        sigma=sigma,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        device=device
    )

    # c) Info-Map
    info_map = get_info(img_t, img_t.shape[0], device=device)

    # d) Normalisierung der beiden Maps
    edge_map = torch.where(edge_map < 0, 0.0, 1.0)
    info_map = -info_map
    info_map = torch.where(info_map < 0, 0.0, 1.0)

    # e) Kombinieren und 3-Kanal-Format herstellen
    combined = edge_map + info_map
    combined = torch.cat([combined, combined, combined], dim=1)  # (1, 3, H, W)

    # f) TPS-Transformation
    combined = TPS_Batch(combined, device=device).detach().to(device)

    # g) Noise z erzeugen
    batch_size = combined.shape[0]  # sollte 1 sein
    z = torch.randn(batch_size, 100, 1, 1, device=device)

    # h) Blur Images erzeugen
    if re12_transform is None or re32_transform is None:
        re12_transform = transforms.Resize((12, 12))
        re32_transform = transforms.Resize((32, 32))

    blur_img = re12_transform(img_t)
    blur_img = re32_transform(blur_img)

    # i) Generator anrufen
    with torch.no_grad():
        out_t = netG(z, combined, blur_img)

    # j) Rückkonvertierung
    out_t = out_t.squeeze(0).cpu().clamp(0.0, 1.0)
    to_pil = transforms.ToPILImage()
    out_img = to_pil(out_t)
    return out_img

if __name__ == "__main__":
    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the image transformations
    re12 = transforms.Resize((12, 12))
    re32 = transforms.Resize((32, 32))

    # Load the generator model
    netG = load_generator("./Result/cifar_gan/tuned_G_16.pth", device)

    # Load the image
    img = Image.open("./src/Baum.jpg").convert("RGB")
    img = re32(img)  # Resize to 32x32

    # Deform the image
    out_img = deform_image(netG, img, device=device, re12_transform=re12, re32_transform=re32)

    # Show the output image
    out_img.show()
