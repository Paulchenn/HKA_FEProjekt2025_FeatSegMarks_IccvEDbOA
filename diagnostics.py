# diagnostics.py
import os, json, argparse, math, time
from typing import Optional, Dict, Any

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# --- Dein Projekt: Imports (Generator, Config, Loader, TIMSE) ---
from utils.helpers import get_config, get_train_val_loaders
from models import generation_imageNet_V2_2 as generation_imageNet

# TIMSE ist optional – wenn Import scheitert, nutzen wir einen Sobel-Backup
try:
    from utils.modules import TIMSE  # erwartet: timse = TIMSE(config); timse.diff_edge_map(img)
    HAS_TIMSE = True
except Exception:
    HAS_TIMSE = False


# ------------------------------
# Util: Device + Fixierbares z
# ------------------------------
def fixed_noise(batch_size, z_dim, device, seed=1234):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((batch_size, z_dim, 1, 1), generator=g, device=device)


# --------------------------------------
# Differenzierbarer Sobel-Edge-Extractor
# --------------------------------------
class SobelEdges(nn.Module):
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]]).view(1, 1, 3, 3)
        gy = torch.tensor([[1., 2., 1.],
                           [0., 0., 0.],
                           [-1., -2., -1.]]).view(1, 1, 3, 3)
        self.register_buffer("gx", gx)
        self.register_buffer("gy", gy)

    def forward(self, x):  # x: (B,C,H,W); Wertebereich (-1..1) oder (0..1)
        # in Graustufen (luminance)
        if x.shape[1] == 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            gray = x
        gx = F.conv2d(gray, self.gx, padding=1)
        gy = F.conv2d(gray, self.gy, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        # normiere grob in [0,1]
        mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return mag


# ------------------------------------
# Down/Up "Blur" (GPU, differentiable)
# ------------------------------------
def blur_by_down_up(x, down_size: int):
    # x in (-1..1); wir interpolieren bilinear
    B, C, H, W = x.shape
    down = F.interpolate(x, size=(down_size, down_size), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
    return up


# -----------------------------------------
# Diagnose-Losses (Edge & Texture separat)
# -----------------------------------------
class EdgeTextureDiagLoss(nn.Module):
    def __init__(self, w_edge=1.0, w_tex=1.0):
        super().__init__()
        self.sobel = SobelEdges()
        self.l1 = nn.L1Loss()
        self.w_edge = float(w_edge)
        self.w_tex = float(w_tex)

    def edge_loss(self, out_img, E_target):
        # bring E_target in [0,1] falls im Bereich (-1..1)
        if E_target.min() < 0:
            E_t = (E_target + 1.0) * 0.5
        else:
            E_t = E_target
        E_out = self.sobel(out_img)
        # falls E_target 3-kanalig: auf grau reduzieren
        if E_t.shape[1] == 3:
            r, g, b = E_t[:, 0:1], E_t[:, 1:2], E_t[:, 2:3]
            E_t = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # auf gleiche Größe bringen
        if E_t.shape[-2:] != E_out.shape[-2:]:
            E_t = F.interpolate(E_t, size=E_out.shape[-2:], mode="nearest")
        return self.l1(E_out, E_t)

    def tex_loss(self, out_img, I_txt):
        # Beide im selben Maßstab; Projekt nutzt (-1..1)
        if out_img.shape != I_txt.shape:
            I_txt = F.interpolate(I_txt, size=out_img.shape[-2:], mode="bilinear", align_corners=False)
        return self.l1(out_img, I_txt)

    def forward(self, out_img, E_target, I_txt):
        L_e = self.edge_loss(out_img, E_target) if self.w_edge > 0 else 0.0 * out_img.mean()
        L_t = self.tex_loss(out_img, I_txt)    if self.w_tex > 0 else 0.0 * out_img.mean()
        # WICHTIG: NICHT detachen – wir brauchen den Graph für autograd.grad(...)
        return self.w_edge * L_e + self.w_tex * L_t, L_e, L_t



# -----------------------------------------
# Forward + Grad-Attribution auf einer Batch
# -----------------------------------------
def grad_attribution(netG, z, E, I_txt, loss_fn: EdgeTextureDiagLoss):
    netG.eval()
    E = E.detach().requires_grad_(True)
    I = I_txt.detach().requires_grad_(True)
    out = netG(z, E, I)
    L_total, L_edge, L_tex = loss_fn(out, E, I)
    gE, gI = torch.autograd.grad(L_edge, [E, I], retain_graph=True, create_graph=False, allow_unused=True)
    if gE is None:
        gE = torch.zeros_like(E)
    if gI is None:
        gI = torch.zeros_like(I)
    # Zusätzlich Grad aus dem Texture-Loss (optional informativ)
    gE_tex, gI_tex = torch.autograd.grad(L_tex, [E, I], retain_graph=False, create_graph=False, allow_unused=True)
    if gE_tex is None:
        gE_tex = torch.zeros_like(E)
    if gI_tex is None:
        gI_tex = torch.zeros_like(I)

    def l2_batch(t):  # (B,C,H,W)
        return torch.sqrt((t ** 2).sum(dim=(1, 2, 3)) + 1e-12)

    stats = {
        "L_total": float(L_total.detach().mean().item()),
        "L_edge": float(L_edge.mean().item()),
        "L_tex": float(L_tex.mean().item()),
        "||dL_edge/dE||": float(l2_batch(gE.detach()).mean().item()),
        "||dL_edge/dI||": float(l2_batch(gI.detach()).mean().item()),
        "||dL_tex/dE||": float(l2_batch(gE_tex.detach()).mean().item()),
        "||dL_tex/dI||": float(l2_batch(gI_tex.detach()).mean().item()),
    }
    return stats


# -----------------------------------------
# Ablation: E→0 und I_txt→0 (Edge/Texture)
# -----------------------------------------
@torch.no_grad()
def ablation_tests(netG, z, E, I_txt, loss_fn: EdgeTextureDiagLoss):
    netG.eval()
    out_base = netG(z, E, I_txt)
    L_base, L_edge_base, L_tex_base = loss_fn(out_base, E, I_txt)

    out_E0 = netG(z, torch.zeros_like(E), I_txt)
    L_E0, L_edge_E0, L_tex_E0 = loss_fn(out_E0, E, I_txt)

    out_T0 = netG(z, E, torch.zeros_like(I_txt))
    L_T0, L_edge_T0, L_tex_T0 = loss_fn(out_T0, E, I_txt)

    return {
        "ΔL_total_E0": float((L_E0 - L_base).mean().item()),
        "ΔL_total_T0": float((L_T0 - L_base).mean().item()),
        "ΔL_edge_E0":  float((L_edge_E0 - L_edge_base).mean().item()),
        "ΔL_edge_T0":  float((L_edge_T0 - L_edge_base).mean().item()),
        "ΔL_tex_E0":   float((L_tex_E0 - L_tex_base).mean().item()),
        "ΔL_tex_T0":   float((L_tex_T0 - L_tex_base).mean().item()),
    }


# -----------------------------------------
# Build E (Edge) und I_txt (blur) pro Batch
# -----------------------------------------
def build_E_and_I(config, img, timse_obj: Optional[Any]):
    # E: bevorzugt TIMSE; Fallback Sobel
    if timse_obj is not None:
        E = timse_obj.diff_edge_map(img)
    else:
        # Sobel auf img (als Pseudo-Edge)
        sob = SobelEdges().to(img.device)
        E = sob(img)            # (B,1,H,W)
        E = E.repeat(1, img.shape[1], 1, 1)  # auf 3 Kanäle broadcasten
        # Skaliere in (-1..1), um das typische Projekt-Range beizubehalten
        E = E * 2.0 - 1.0
    # I_txt: GPU-kompatibles Down/Up
    I_txt = blur_by_down_up(img, config.DOWN_SIZE)
    return E, I_txt


# -----------------------------------------
# Hauptdiagnose über n Batches
# -----------------------------------------
def run_diagnostics(
    config_path: str = "config/config.json",
    ckpt_path: Optional[str] = None,
    split: str = "val",
    n_batches: int = 16,
    seed: int = 1234,
    save_dir_override: Optional[str] = None
) -> Dict[str, float]:
    cfg = get_config(config_path)
    device = cfg.DEVICE

    # Ausgabeverzeichnis direkt relativ zum --ckpt wählen:
    #   ./Result/<RUN_ID>/diagnostics/
    if save_dir_override:
        out_dir = os.path.join(save_dir_override, "diagnostics")
    else:
        ckpt_path_abs = os.path.abspath(args.ckpt)  # z.B. ./Result/20250826_193108/ImageNet256/tuned_netG/netG__stage1__Epoch_023.pth
        ckpt_dir = os.path.dirname(ckpt_path_abs)   # .../ImageNet256/tuned_netG
        # Zwei Ebenen hoch: .../20250826_193108
        run_root = os.path.abspath(os.path.join(ckpt_dir, os.pardir, os.pardir))
        out_dir = os.path.join(run_root, "diagnostics")

    os.makedirs(out_dir, exist_ok=True)


    # Loader aufsetzen – benutze die gleichen Transforms wie im Training
    transform_train = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.CenterCrop(cfg.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.CenterCrop(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Tilde im Pfad sicher expandieren
    data_dir = os.path.expanduser(cfg.DATASET_PATH)

    train_loader, val_loader = get_train_val_loaders(
        cfg,
        data_dir=data_dir,
        transform_train=transform_train,
        transform_val=transform_val,
        pin_memory=True
    )
    loader = val_loader if split == "val" else train_loader


    # Generator
    netG = generation_imageNet.generator(img_size=cfg.IMG_SIZE, z_dim=cfg.NOISE_SIZE).to(device)

    # Checkpoint laden (Stage-1-best empfohlen)
    ckpt_guess = ckpt_path or getattr(cfg, "PATH_TUNED_G", None)
    if ckpt_guess is None or (isinstance(ckpt_guess, str) and ckpt_guess.strip() == ""):
        raise FileNotFoundError("Kein Generator-Checkpoint angegeben. Nutze --ckpt PATH/zur/netG__stage1__Epoch_xxx.pth")
    if not os.path.exists(ckpt_guess):
        print(f"[Warnung] Checkpoint {ckpt_guess} existiert nicht. Bitte --ckpt explizit setzen.")
    else:
        sd = torch.load(ckpt_guess, map_location=device)
        model_dict = netG.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        netG.load_state_dict(model_dict)
        print(f"Loaded netG weights from: {ckpt_guess}")

    # TIMSe optional vorbereiten
    timse_obj = TIMSE(config=cfg) if HAS_TIMSE else None
    if HAS_TIMSE:
        print("[Diag] TIMSE verfügbar – nutze TIMSE.diff_edge_map(img) als Edge-Target.")
    else:
        print("[Diag] TIMSE nicht gefunden – fallback auf Sobel-Edges als Edge-Target.")

    # Loss
    loss_fn = EdgeTextureDiagLoss(w_edge=1.0, w_tex=1.0).to(device)

    # Sammelstatistiken
    agg = []
    torch.set_grad_enabled(True)  # wir arbeiten im eval-Mode, aber Grad für Attribution an

    it = 0
    for img, _ in loader:
        if it >= n_batches:
            break
        img = img.to(device, non_blocking=True)

        # Inputs bauen
        E, I_txt = build_E_and_I(cfg, img, timse_obj=timse_obj)
        z = fixed_noise(img.shape[0], cfg.NOISE_SIZE, device, seed=seed + it)

        # Ablation (ohne Grad)
        ab = ablation_tests(netG, z, E, I_txt, loss_fn)

        # Grad-Attribution (mit Grad)
        ga = grad_attribution(netG, z, E, I_txt, loss_fn)

        entry = {**ab, **ga}
        agg.append(entry)

        # Kurz-Print pro Batch
        print(f"[{it+1}/{n_batches}] "
              f"ΔL_edge(E0/T0)={entry['ΔL_edge_E0']:.4f}/{entry['ΔL_edge_T0']:.4f} | "
              f"Grad edge: ||dL/dE||={entry['||dL_edge/dE||']:.4e}, ||dL/dI||={entry['||dL_edge/dI||']:.4e}")

        it += 1

    # Mittelwerte
    keys = agg[0].keys()
    mean_stats = {k: float(sum(d[k] for d in agg) / len(agg)) for k in keys}

    # Heuristiken zur Einschätzung
    mean_stats["edge_underuse_score"] = float(mean_stats["ΔL_edge_T0"] - mean_stats["ΔL_edge_E0"])
    # <0 → E wichtig; >0 → Texture wichtiger für Edge-Kriterium
    denom = (mean_stats["||dL_edge/dI||"] + 1e-12)
    mean_stats["grad_ratio_edge_E_over_I"] = float(mean_stats["||dL_edge/dE||"] / denom)

    # Speichern unter: ./Result/<RUN_ID>/diagnostics/diagnostic__<CKPT_STEM>.json
    # Beispiel: diagnostic__netG__stage1__Epoch_023.json
    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_json = os.path.join(out_dir, f"diagnostic__{ckpt_stem}.json")

    with open(out_json, "w") as f:
        json.dump({"per_batch": agg, "mean": mean_stats}, f, indent=2)

    print(f"\n[Diag] Zusammenfassung gespeichert: {out_json}")
    print(json.dumps(mean_stats, indent=2))


    # Kurze Interpretation in Textform
    print("\n[Diag] Interpretation (Daumenregel):")
    print(" - Ist edge_underuse_score >> 0  und grad_ratio_edge_E_over_I << 1,")
    print("   dann wird der Edge-Pfad wahrscheinlich unternutzt (Texture dominiert).")
    print(" - Ziel: edge_underuse_score <= 0   und grad_ratio_edge_E_over_I >= 0.5 (grobe Richtwerte).")

    return mean_stats


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Diagnose: Nutzt der Generator die Edge-Map (Stage-1-best)?")
    p.add_argument("--config", type=str, default="config/config.json", help="Pfad zur config.json")
    p.add_argument("--ckpt", type=str, default=None, help="Pfad zu netG__stage1__Epoch_xxx.pth (Best aus Stage 1)")
    p.add_argument("--split", type=str, default="val", choices=["val", "train"], help="Auf welcher Split diagnostizieren")
    p.add_argument("--batches", type=int, default=16, help="Wieviele Batches inspizieren")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--save_dir", type=str, default=None, help="Override für Ausgabeverzeichnis")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    #pdb.set_trace()
    run_diagnostics(
        config_path=args.config,
        ckpt_path=args.ckpt,
        split=args.split,
        n_batches=args.batches,
        seed=args.seed,
        save_dir_override=args.save_dir
    )
