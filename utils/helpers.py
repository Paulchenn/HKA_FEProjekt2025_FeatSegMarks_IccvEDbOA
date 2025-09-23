import json, os, time, pdb, pytz, itertools, glob, os, random

from collections import deque

from contextlib import nullcontext

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

import torch
from torch import amp
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.autograd import Variable

from types import SimpleNamespace


#def preperations():


def blur_image(
    img,
    downSize=12
):
    # img: [B,C,H,W] in [-1,1]
    B, C, H, W = img.shape

    # downsample auf Quadrat (downSize, downSize) ist ok...
    downsampler = transforms.Resize((downSize, downSize))
    downsampled = downsampler(img)

    # ...aber UPSAMPLE MUSS auf Original-HW gehen:
    upsampler = transforms.Resize((H, W))

    return upsampler(downsampled)

def blur_image_dynamic(
    img,
    down_min=64,
    down_max=128,
    p_heavy=0.35,
    heavy_down=12,
    p_zero=0.15,
    noise_std=0.0,
):
    if torch.rand(1).item() < p_zero:
        return torch.zeros_like(img)
    if torch.rand(1).item() < p_heavy:
        downSize = heavy_down
    else:
        downSize = int(torch.randint(low=down_min, high=down_max + 1, size=(1,)).item())

    B, C, H, W = img.shape
    out = transforms.Resize((H, W))(transforms.Resize((downSize, downSize))(img))
    if noise_std > 0:
        noise = torch.randn_like(out) * noise_std
        out = torch.clamp(out + noise, -1.0, 1.0)
    return out



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
    config["PATH_OPTIM_D_S1"] = config["path_optim_D_stage1"]
    config["PATH_OPTIM_D_S2"] = config["path_optim_D_stage2"]
    config["PATH_OPTIM_G_S1"] = config["path_optim_G_stage1"]
    config["PATH_OPTIM_G_S2"] = config["path_optim_G_stage2"]

    config["NOISE_SIZE"] = config["noise_size"]

    config["PATIENCE_S1"] = config["patience_stage1"]
    config["PATIENCE_S2"] = config["patience_stage2"]

    config["SHOW_IMG_EMSE_TSD"] = config["show_imgEmseTsd"]
    config["SHOW_TSG"] = config["show_tsg"]
    config["SHOW_IMAGES_INTERVAL"] = config["show_images_interval"]

    config["LOG_INTERVAL"] = config["log_interval"]

    config["TRAIN_WITH_CLS"] = config["train_with_cls"]

    config["DOWN_SIZE"] = config["downSize"]
    config["DOWN_SIZE2"] = config["downSize2"]

    config["DEBUG_MODE"] = config["debug_mode"]
    config["DEBUG_ITERS_START"] = config["debugIterations_strt"]
    config["DEBUG_ITERS_AMOUNT"] = config["debugIterations_amount"]

    config["LR_D_S1"] = config["learning_rate"]["discriminator_stage1"]
    config["LR_D_S2"] = config["learning_rate"]["discriminator_stage2"]
    config["LR_G_S1"] = config["learning_rate"]["generator_stage1"]
    config["LR_G_S2"] = config["learning_rate"]["generator_stage2"]

    config["LAMBDA_S1_L1"]      = config["loss_weights"]["lambda_stage1_l1"]
    config["LAMBDA_S1_GAN"]     = config["loss_weights"]["lambda_stage1_GAN"]
    config["LAMBDA_S1_CE"]      = config["loss_weights"]["lambda_stage1_CE"]
    config["LAMBDA_S1_EDGE"]    = config["loss_weights"]["lambda_stage1_edge"]

    config["LAMBDA_S2_L1"]      = config["loss_weights"]["lambda_stage2_l1"]
    config["LAMBDA_S2_GAN"]     = config["loss_weights"]["lambda_stage2_GAN"]
    config["LAMBDA_S2_CE"]      = config["loss_weights"]["lambda_stage2_CE"]
    config["LAMBDA_S2_CLS"]     = config["loss_weights"]["lambda_stage2_cls"]
    config["LAMBDA_S2_EDGE"]    = config["loss_weights"]["lambda_stage2_edge"]

    # Dataset (robust for MegaDepth)
    ds = config.get("dataset", {})
    config["DATASET_NAME"] = ds.get("name", "image_only")
    config["DATASET_PATH"] = ds.get("path", "")
    # ImageNet-spezifisch nur, wenn vorhanden:
    if "SELECTED_SYNSETS" in ds:
        config["SELECTED_SYNSETS"] = ds["SELECTED_SYNSETS"]
        config["CLASS_NAMES"] = list(ds["SELECTED_SYNSETS"].values())
        config["NUM_CLASSES"] = len(config["CLASS_NAMES"])
    else:
        config["CLASS_NAMES"] = []
        # Wenn du ohne Klassifikation trainierst, nimm 1 als Dummy:
        config["NUM_CLASSES"] = int(config.get("num_classes", 1))
        config["TRAIN_WITH_CLS"] = bool(config.get("train_with_cls", False))

    # Set timezone (z. B. Europe/Berlin)
    tz = pytz.timezone('Europe/Berlin')
    today_str = datetime.now(tz).strftime('%Y%m%d_%H%M%S')
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
        class_to_idx = base_dataset.class_to_idx
        selected_idx = [class_to_idx[name] for name in class_names if name in class_to_idx]

        # Filter samples
        filtered_samples = [s for s in base_dataset.samples if s[1] in selected_idx]

        # Remap class indices to 0-based
        idx_to_keep = sorted(set(s[1] for s in filtered_samples))
        idx_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(idx_to_keep)}

        # Update samples and targets
        base_dataset.samples = [(s[0], idx_remap[s[1]]) for s in filtered_samples]
        base_dataset.targets = [idx_remap[s[1]] for s in filtered_samples]

        # Update class_to_idx and classes
        base_dataset.class_to_idx = {name: idx_remap[class_to_idx[name]] for name in class_names if name in class_to_idx}
        base_dataset.classes = list(base_dataset.class_to_idx.keys())

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
    
    # Debug output
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")


    img_train, label_train = next(iter(train_loader))
    img_val, label_val = next(iter(val_loader))

    print("Train batch shape:", img_train.shape, "Labels:", label_train)
    print("Val batch shape:", img_val.shape, "Labels:", label_val)

    print("Number of images in train_loader:", len(train_loader.dataset))
    print("Number of classes in train_loader:", len(train_loader.dataset.dataset.classes))
    print("Number of images in val_loader:", len(val_loader.dataset))
    print("Number of classes in val_loader:", len(val_loader.dataset.dataset.classes))

    return train_loader, val_loader


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

class ImageOnlyDataset(Dataset):
    def __init__(self, root):
        self.paths = []
        for ext in IMG_EXTS:
            self.paths.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
        self.paths.sort()
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under {root}")

    def __len__(self): return len(self.paths)

    def get_path(self, idx):  # schlanker Zugriff für Views
        return self.paths[idx]

class ImageView(Dataset):
    """Ein View auf ImageOnlyDataset mit eigener Transform und eigenen Indices."""
    def __init__(self, base_ds: ImageOnlyDataset, indices, transform=None):
        self.base = base_ds
        self.indices = list(indices)
        self.transform = transform

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        p = self.base.get_path(self.indices[i])
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, 0  # Dummy-Label

def build_image_only_loaders(config, transform_train, transform_val, pin_memory=True):
    debug_prints = False

    # Pfad (dict oder Namespace – beide Varianten unterstützen)
    dataset_path = config.dataset["path"] if isinstance(config.dataset, dict) else config.dataset.path
    base = ImageOnlyDataset(dataset_path)

    use_ratio = getattr(config, "use_ratio", 0.01)
    val_ratio = getattr(config, "val_ratio", 0.2)

    n_total = len(base)
    n_use   = max(1, int(round(n_total * use_ratio)))
    n_val   = max(1, int(round(n_use * val_ratio)))
    n_train = max(1, n_use - n_val)

    # Reproduzierbare, zufällige Auswahl von n_use Indices
    g = torch.Generator().manual_seed(42)
    all_idx = torch.randperm(n_total, generator=g).tolist()
    use_idx = all_idx[:n_use]

    # Splitten in Train/Val
    train_idx = use_idx[:n_train]
    val_idx   = use_idx[n_train:n_train + n_val]
    if len(val_idx) == 0:  # Sicherheitsnetz
        val_idx = use_idx[-1:]
        train_idx = use_idx[:-1]

    # Eigene Views mit EIGENER Transform
    train_ds = ImageView(base, train_idx, transform=transform_train)
    val_ds   = ImageView(base, val_idx,   transform=transform_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=getattr(config, "BATCH_SIZE", 16),
        shuffle=True,
        num_workers=getattr(config, "NUM_WORKERS", 8),
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=getattr(config, "BATCH_SIZE", 16),
        shuffle=False,
        num_workers=getattr(config, "NUM_WORKERS", 8),
        pin_memory=pin_memory,
        drop_last=False,
    )

    if debug_prints:
        print(f"[LOADER(img-only)] Dataset path: {dataset_path}")
        print(f"[LOADER(img-only)] n_total: {n_total}")
        print(f"[LOADER(img-only)] n_use:   {n_use}  (ratio={use_ratio})")
        print(f"[LOADER(img-only)] n_train: {len(train_ds)}")
        print(f"[LOADER(img-only)] n_val:   {len(val_ds)}")
        # Optional: erste paar Pfade für Kontrolle
        # print([base.get_path(i) for i in train_idx[:3]])

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

def _move_fig_top_right(fig):
    """Place the window so that its top-right corner touches the screen's top-right corner."""
    backend = matplotlib.get_backend().lower()
    mgr = fig.canvas.manager
    try:
        # ---- Qt (Qt5Agg / QtAgg) ----
        if "qt" in backend:
            win = mgr.window
            win.show()                      # ensure window exists and has a size
            fig.canvas.draw_idle()
            plt.pause(0.05)

            rect = win.frameGeometry()      # includes window decorations
            ww, wh = rect.width(), rect.height()

            scr = win.screen().availableGeometry()
            x = scr.right() - ww            # right edge minus window width
            y = scr.top()                   # top edge
            win.move(x, y)

        # ---- GTK3 (Gtk3Agg) ----
        elif "gtk3" in backend or "gtk" in backend:
            # Requires PyGObject (should be present if GTK3Agg works)
            import gi
            gi.require_version("Gtk", "3.0")
            gi.require_version("Gdk", "3.0")
            from gi.repository import Gtk, Gdk

            win = mgr.window                 # Gtk.Window
            win.show_all()

            # process pending events so sizes are realized
            while Gtk.events_pending():
                Gtk.main_iteration_do(False)

            gdk_win = win.get_window()
            if gdk_win is None:
                return

            # window size including frame (decorations)
            frame = gdk_win.get_frame_extents()
            ww, wh = frame.width, frame.height

            # preferred: monitor workarea; fallback: screen size
            x = y = 0
            # try:
            #     display = Gdk.Display.get_default()
            #     monitor = display.get_primary_monitor()
            #     # workarea excludes panels/docks if available
            #     if hasattr(monitor, "get_workarea"):
            #         area = monitor.get_workarea()
            #     else:
            #         area = monitor.get_geometry()
            #     sw, sh = area.width, area.height
            #     sx, sy = area.x, area.y
            #     x = sx + sw - ww
            #     y = sy
            # except Exception:
            screen = win.get_screen()
            sw, sh = screen.get_width(), screen.get_height()
            x = sw - ww
            y = 0

            # On Wayland this may be ignored by the compositor
            win.move(int(x), int(y))

        # ---- Tk (TkAgg) ----
        elif "tk" in backend:
            win = mgr.window
            win.update_idletasks()
            ww, wh = win.winfo_width(), win.winfo_height()
            sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
            x = sw - ww
            y = 0
            win.wm_geometry(f"+{x}+{y}")

        else:
            print(f"[show_tsg] Window move not supported for backend '{backend}'.")
    except Exception as e:
        print(f"[show_tsg] Could not move window: {e}")



def _to_numpy_img(t: torch.Tensor):
    """
    Erwartet Tensor [B,C,H,W] oder [C,H,W] in [-1,1].
    Gibt float32-Numpy-Bild [H,W,3] in [0,1] zurück.
    """
    if t.dim() == 4:
        t = t[0]  # erstes Bild
    if t.size(0) == 1:
        t = t.repeat(3, 1, 1)  # Grau -> RGB

    # WICHTIG: float() zwingt fp32 (verhindert Matplotlib-Fehler)
    img = t.detach().float().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 0.5 + 0.5, 0.0, 1.0)
    return img.astype(np.float32)


def show_tsg(
    img=None,
    e_extend=None,
    e_deformed=None,
    img_blur=None,
    img_for_loss=None,
    G_rough=None,
    G_fine=None,
    G_fine_resized=None,
    G_fine_norm=None,
    e_extend_G_fine=None,
    e_edeformed_G_fine=None
):
    """
    Visualize inputs/outputs for TSG: originals, edges, blurred image, and generated results.
    """
    fig = plt.figure(figsize=(10.5, 8))  # create the figure first (we will move the window later)

    # === show original and derived images ===
    try:
        plt.subplot(3, 4, 1); plt.title('original image'); plt.imshow(_to_numpy_img(img))
    except: pass
    try:
        plt.subplot(3, 4, 2); plt.title('edge map');       plt.imshow(_to_numpy_img(e_extend))
    except: pass
    try:
        plt.subplot(3, 4, 3); plt.title('deformed edge');  plt.imshow(_to_numpy_img(e_deformed))
    except: pass
    try:
        plt.subplot(3, 4, 4); plt.title('blured image');   plt.imshow(_to_numpy_img(img_blur))
    except: pass
    try:
        plt.subplot(3, 4, 5); plt.title('image for loss'); plt.imshow(_to_numpy_img(img_for_loss))
    except: pass
    try:
        plt.subplot(3, 4, 6); plt.title('gen img s1');     plt.imshow(_to_numpy_img(G_rough))
    except: pass
    try:
        plt.subplot(3, 4, 7); plt.title('gen img s2');     plt.imshow(_to_numpy_img(G_fine))
    except: pass
    try:
        plt.subplot(3, 4, 8); plt.title('resized img s2'); plt.imshow(_to_numpy_img(G_fine_resized))
    except: pass
    try:
        plt.subplot(3, 4, 9); plt.title('norm res img s2');plt.imshow(_to_numpy_img(G_fine_norm))
    except: pass
    try:
        plt.subplot(3, 4, 10);plt.title('edge gen s2');    plt.imshow(_to_numpy_img(e_extend_G_fine))
    except: pass
    try:
        plt.subplot(3, 4, 11);plt.title('def gen s2');     plt.imshow(_to_numpy_img(e_edeformed_G_fine))
    except: pass

    plt.tight_layout()
    plt.show(block=False)       # show the window (non-blocking)
    _move_fig_top_right(fig)    # move it to the top-right corner
    plt.pause(0.1)              # keep it on screen briefly (adjust as needed)
    plt.close(fig)



def show_result(
        config,
        num_epoch,
        img=None,
        edgeMap=None,
        path='Result/result.png',
        print_original=False,
        show=False,
        save=False,
        training_stage=1,
        *,
        netG
):
    # neue Autocast-API (oder weglassen; unten casten wir sowieso auf fp32 fürs Plotten)
    autocast_ctx = torch.amp.autocast("cuda") if config.DEVICE.type != "cpu" else nullcontext()

    mn_batch = edgeMap.shape[0]

    if not print_original:
        # Nur fürs Sampling: eval + inference_mode spart Speicher/grad
        netG.eval()
        with torch.inference_mode(), autocast_ctx:
            zdim = getattr(config, "noise_size", 100)
            z_ = torch.randn((mn_batch, zdim), device=config.DEVICE).view(-1, zdim, 1, 1)
            img_blur = blur_image(img, config.DOWN_SIZE if training_stage == 1 else config.DOWN_SIZE2)
            test_images = netG(z_, edgeMap, img_blur)
        netG.train()
        myTitle = 'Generated Images'
    else:
        test_images = img
        myTitle = 'Original Images'

    # >>> WICHTIG: vor dem Plotten in FP32 auf CPU bringen <<<
    test_images = test_images.detach().float().cpu()  # <-- verhindert Matplotlib-Error

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
            arr = test_images[k].numpy().transpose(1, 2, 0)  # [H,W,C]
            arr = np.clip(arr * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)  # sicher float32
            ax[i, j].imshow(arr)

    label = f'Epoch {num_epoch+1}'
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
        plt.close(fig)


