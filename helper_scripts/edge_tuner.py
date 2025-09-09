#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge Tuner (τ + optionale Morphologie/Blur) für DS_EMSE:
- Lädt Config (DS_TAU, DS_BETA, DS_FILTER_METHOD, DS_PRE_BLUR_K, DS_OPEN_KS, DS_CLOSE_KS, optional IMG_SIZE)
- τ/β-Slider, Filter-Dropdown, Kernel-Slider
- Pipeline: (Pre-Blur) -> Sobel -> Sigmoid(β*(mag-τ)) -> (Open/Close) -> invertiert -> Anzeige
- Buttons: Bild öffnen, Werte speichern, Kantenkarte exportieren
"""

import argparse
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6
FILTER_CHOICES = ["None", "Open", "Close", "Open+Close", "Close+Open"]

# ------------------------------- Config I/O -------------------------------

def load_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config nicht gefunden: {path}")
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg

def save_config(path, cfg):
    # Bestehende Config beibehalten und nur relevante Keys updaten
    try:
        with open(path, "r") as f:
            out = json.load(f)
    except Exception:
        out = {}

    out["DS_TAU"]           = float(cfg.get("DS_TAU", 0.30))
    out["DS_BETA"]          = float(cfg.get("DS_BETA", 25.0))
    out["DS_FILTER_METHOD"] = str(cfg.get("DS_FILTER_METHOD", "None"))
    out["DS_PRE_BLUR_K"]    = int(cfg.get("DS_PRE_BLUR_K", 0))
    out["DS_OPEN_KS"]       = int(cfg.get("DS_OPEN_KS", 0))
    out["DS_CLOSE_KS"]      = int(cfg.get("DS_CLOSE_KS", 0))

    if "IMG_SIZE" in cfg:
        out["IMG_SIZE"] = cfg["IMG_SIZE"]

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out

# ------------------------------- Bild & Tensor -------------------------------

def pil_to_tensor(img_pil, img_size=256):
    """PIL RGB -> torch.Tensor [1,3,H,W] in [-1,1]"""
    if isinstance(img_size, (list, tuple)):
        size = (int(img_size[0]), int(img_size[1]))
    else:
        size = (int(img_size), int(img_size))
    img_pil = img_pil.convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(img_pil).astype(np.float32) / 255.0   # [H,W,3] 0..1
    arr = arr * 2.0 - 1.0                                  # [-1,1]
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return ten

# ------------------------------- Kern-Helfer -------------------------------

def _odd_or_zero(k: int) -> int:
    """0 bleibt 0 (deaktiviert). k>=1 wird auf ungerade korrigiert."""
    k = int(max(0, k))
    if k == 0:
        return 0
    return k if (k % 2 == 1) else (k + 1)

def _box_blur(gray: torch.Tensor, k: int) -> torch.Tensor:
    """Einfacher Box-Blur auf [1,1,H,W]; k=0 → passthrough."""
    k = _odd_or_zero(k)
    if k <= 1:
        return gray
    w = torch.ones((1, 1, k, k), dtype=gray.dtype, device=gray.device) / float(k * k)
    return F.conv2d(gray, w, padding=k // 2)

# ------------------------------- Morphologie (auf s: Kanten hell) -------------------------------

def _dilate(x: torch.Tensor, ks: int) -> torch.Tensor:
    """Max-Filter, erhält Form [1,1,H,W]"""
    ks = _odd_or_zero(ks)
    if ks <= 1:
        return x
    return F.max_pool2d(x, kernel_size=ks, stride=1, padding=ks // 2)

def _erode(x: torch.Tensor, ks: int) -> torch.Tensor:
    """Min-Filter via -MaxPool, erhält Form [1,1,H,W]"""
    ks = _odd_or_zero(ks)
    if ks <= 1:
        return x
    return -F.max_pool2d(-x, kernel_size=ks, stride=1, padding=ks // 2)

def _open(x: torch.Tensor, ks: int) -> torch.Tensor:
    return _dilate(_erode(x, ks), ks)

def _close(x: torch.Tensor, ks: int) -> torch.Tensor:
    return _erode(_dilate(x, ks), ks)

def apply_filter_s(s: torch.Tensor, method: str, ks_open: int, ks_close: int) -> torch.Tensor:
    """Morphologie auf s (Kanten hell). method bestimmt Reihenfolge."""
    method = (method or "None").strip()
    if method not in FILTER_CHOICES:
        method = "None"

    if method == "None":
        return s
    elif method == "Open":
        return _open(s, ks_open)
    elif method == "Close":
        return _close(s, ks_close)
    elif method == "Open+Close":
        return _close(_open(s, ks_open), ks_close)
    elif method == "Close+Open":
        return _open(_close(s, ks_close), ks_open)
    return s

# ------------------------------- Kantenberechnung -------------------------------

def tensor_edge_with_filters(
    img, tau=0.30, beta=25.0,
    filter_method="None",
    pre_blur_k=0, open_ks=0, close_ks=0,
    eps=EPS
):
    """
    img: torch [1,3,H,W] in [-1,1]
    Rückgabe: torch [H,W] in [0,1], „Hintergrund hell, Kanten dunkel“ (invertiert)
    """
    B, C, H, W = img.shape
    assert B == 1, "Dieses Tool rechnet mit einem Einzelbild."

    # Grauwert
    if C == 3:
        r = img[:, 0:1]; g = img[:, 1:2]; b = img[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = img

    # (optional) Pre-Blur vor der Ableitung
    gray = _box_blur(gray, pre_blur_k)

    # Sobel
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + eps)  # [1,1,H,W]

    # pro Bild normalisieren
    mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + eps)

    # τ/β-Gating: starke Kanten ~1 (hell)
    s = torch.sigmoid(float(beta) * (mag - float(tau)))  # [1,1,H,W]

    # (optional) Morphologie auf s
    s = apply_filter_s(s, filter_method, open_ks, close_ks)

    # invertiert (für deine Pipeline: Hintergrund hell)
    inv = 1.0 - s
    inv = inv.squeeze(0).squeeze(0)  # [H,W]
    inv = torch.clamp(inv, 0.0, 1.0)
    return inv

# ------------------------------- Anzeige-Helfer -------------------------------

def to_photoimage(gray_01):
    """torch/numpy [H,W] (0..1) -> PhotoImage (PIL L)"""
    if isinstance(gray_01, torch.Tensor):
        arr = gray_01.detach().cpu().numpy()
    else:
        arr = np.asarray(gray_01)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(arr, mode="L")
    return ImageTk.PhotoImage(pil)

def side_by_side(original_pil, edge_gray_01):
    """Original links, Kanten rechts."""
    arr = edge_gray_01.detach().cpu().numpy() if isinstance(edge_gray_01, torch.Tensor) else np.asarray(edge_gray_01)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    edge_pil = Image.fromarray(arr, mode="L").convert("RGB")
    W, H = original_pil.size
    edge_pil = edge_pil.resize((W, H), Image.NEAREST)
    combo = Image.new("RGB", (W * 2, H))
    combo.paste(original_pil, (0, 0)); combo.paste(edge_pil, (W, 0))
    return combo

# ------------------------------- GUI -------------------------------

class EdgeTunerApp:
    def __init__(self, master, cfg_path, img_path):
        self.master = master
        self.cfg_path = cfg_path
        self.cfg = load_config(cfg_path)

        # Defaults
        self.tau   = float(self.cfg.get("DS_TAU", 0.30))
        self.beta  = float(self.cfg.get("DS_BETA", 25.0))
        self.filt  = str(self.cfg.get("DS_FILTER_METHOD", "None"))
        self.blurK = int(self.cfg.get("DS_PRE_BLUR_K", 0))
        self.openK = int(self.cfg.get("DS_OPEN_KS", 0))
        self.closeK= int(self.cfg.get("DS_CLOSE_KS", 0))

        img_size  = self.cfg.get("IMG_SIZE", 256)
        self.img_size = int(img_size if isinstance(img_size, int) else img_size[0])

        # Bild laden
        self.original_pil = self._load_image(img_path)
        self.img_tensor = pil_to_tensor(self.original_pil, self.img_size)

        master.title("DS_EMSE Edge Tuner (τ + Morphologie)")
        master.geometry("1120x780")

        # --- Image area
        self.img_label = tk.Label(master)
        self.img_label.pack(side=tk.TOP, padx=8, pady=8)

        # --- Controls
        controls = tk.Frame(master)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        # τ
        tk.Label(controls, text="τ (Schwelle)").grid(row=0, column=0, sticky="w")
        self.tau_scale = tk.Scale(controls, from_=0, to=100, orient=tk.HORIZONTAL,
                                  command=self.on_change, length=420, resolution=1)
        self.tau_scale.set(int(round(self.tau * 100)))
        self.tau_scale.grid(row=0, column=1, sticky="we", padx=6)
        self.tau_value_lbl = tk.Label(controls, text=f"{self.tau:.2f}")
        self.tau_value_lbl.grid(row=0, column=2, sticky="w")

        # β
        tk.Label(controls, text="β (Steilheit)").grid(row=1, column=0, sticky="w")
        self.beta_scale = tk.Scale(controls, from_=1, to=60, orient=tk.HORIZONTAL,
                                   command=self.on_change, length=420, resolution=1)
        self.beta_scale.set(int(round(self.beta)))
        self.beta_scale.grid(row=1, column=1, sticky="we", padx=6)
        self.beta_value_lbl = tk.Label(controls, text=f"{self.beta:.0f}")
        self.beta_value_lbl.grid(row=1, column=2, sticky="w")

        # Filtermethode Dropdown
        tk.Label(controls, text="Filtermethode").grid(row=2, column=0, sticky="w")
        self.filter_var = tk.StringVar(value=self.filt if self.filt in FILTER_CHOICES else "None")
        self.filter_menu = tk.OptionMenu(controls, self.filter_var, *FILTER_CHOICES, command=lambda _=None: self.on_change())
        self.filter_menu.config(width=16)
        self.filter_menu.grid(row=2, column=1, sticky="w", padx=6)
        tk.Label(controls, text="(Reihenfolge wirkt auf s=Kanten hell)").grid(row=2, column=2, sticky="w")

        # Pre-Blur K
        tk.Label(controls, text="Pre-Blur K (0=aus)").grid(row=3, column=0, sticky="w")
        self.blur_scale = tk.Scale(controls, from_=0, to=31, orient=tk.HORIZONTAL,
                                   command=self.on_change, length=420, resolution=1)
        self.blur_scale.set(self.blurK)
        self.blur_scale.grid(row=3, column=1, sticky="we", padx=6)
        self.blur_value_lbl = tk.Label(controls, text=f"effektiv: {_odd_or_zero(self.blurK)}")
        self.blur_value_lbl.grid(row=3, column=2, sticky="w")

        # Open KS
        tk.Label(controls, text="Open KS (0=aus)").grid(row=4, column=0, sticky="w")
        self.open_scale = tk.Scale(controls, from_=0, to=31, orient=tk.HORIZONTAL,
                                   command=self.on_change, length=420, resolution=1)
        self.open_scale.set(self.openK)
        self.open_scale.grid(row=4, column=1, sticky="we", padx=6)
        self.open_value_lbl = tk.Label(controls, text=f"effektiv: {_odd_or_zero(self.openK)}")
        self.open_value_lbl.grid(row=4, column=2, sticky="w")

        # Close KS
        tk.Label(controls, text="Close KS (0=aus)").grid(row=5, column=0, sticky="w")
        self.close_scale = tk.Scale(controls, from_=0, to=31, orient=tk.HORIZONTAL,
                                    command=self.on_change, length=420, resolution=1)
        self.close_scale.set(self.closeK)
        self.close_scale.grid(row=5, column=1, sticky="we", padx=6)
        self.close_value_lbl = tk.Label(controls, text=f"effektiv: {_odd_or_zero(self.closeK)}")
        self.close_value_lbl.grid(row=5, column=2, sticky="w")

        # --- Buttons
        btns = tk.Frame(master)
        btns.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        tk.Button(btns, text="Bild öffnen…", command=self.choose_image).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Werte in Config speichern", command=self.save_values).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Kantenkarte exportieren", command=self.export_edge).pack(side=tk.LEFT, padx=4)

        # Initiales Rendern
        self.render()

    def _load_image(self, path):
        if path and os.path.isfile(path):
            return Image.open(path).convert("RGB")
        else:
            messagebox.showinfo("Bild wählen", "Bitte ein Bild auswählen.")
            p = filedialog.askopenfilename(filetypes=[("Bilder", "*.png;*.jpg;*.jpeg;*.bmp")])
            if not p:
                raise RuntimeError("Kein Bild gewählt.")
            return Image.open(p).convert("RGB")

    def on_change(self, _evt=None):
        self.tau   = float(self.tau_scale.get()) / 100.0
        self.beta  = float(self.beta_scale.get())
        self.filt  = self.filter_var.get()
        self.blurK = int(self.blur_scale.get())
        self.openK = int(self.open_scale.get())
        self.closeK= int(self.close_scale.get())

        self.tau_value_lbl.config(text=f"{self.tau:.2f}")
        self.beta_value_lbl.config(text=f"{self.beta:.0f}")
        self.blur_value_lbl.config(text=f"effektiv: {_odd_or_zero(self.blurK)}")
        self.open_value_lbl.config(text=f"effektiv: {_odd_or_zero(self.openK)}")
        self.close_value_lbl.config(text=f"effektiv: {_odd_or_zero(self.closeK)}")

        self.render()

    def _compute_edge(self):
        with torch.no_grad():
            edge_inv = tensor_edge_with_filters(
                self.img_tensor,
                tau=self.tau,
                beta=self.beta,
                filter_method=self.filt,
                pre_blur_k=self.blurK,
                open_ks=self.openK,
                close_ks=self.closeK
            )
        return edge_inv

    def render(self):
        edge = self._compute_edge()
        disp_orig = self.original_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        combo = side_by_side(disp_orig, edge)
        self._combo_imgtk = ImageTk.PhotoImage(combo)  # Referenz halten!
        self.img_label.config(image=self._combo_imgtk)

    def choose_image(self):
        p = filedialog.askopenfilename(filetypes=[("Bilder", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not p:
            return
        try:
            self.original_pil = Image.open(p).convert("RGB")
            self.img_tensor = pil_to_tensor(self.original_pil, self.img_size)
            self.render()
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Bild nicht laden:\n{e}")

    def save_values(self):
        try:
            cfg = dict(self.cfg)
            cfg["DS_TAU"]           = float(self.tau)
            cfg["DS_BETA"]          = float(self.beta)
            cfg["DS_FILTER_METHOD"] = str(self.filt)
            cfg["DS_PRE_BLUR_K"]    = int(self.blurK)
            cfg["DS_OPEN_KS"]       = int(self.openK)
            cfg["DS_CLOSE_KS"]      = int(self.closeK)
            save_config(self.cfg_path, cfg)
            messagebox.showinfo(
                "Gespeichert",
                f"DS_TAU={self.tau:.2f}, DS_BETA={self.beta:.0f}\n"
                f"Filter={self.filt}, PreBlurK={self.blurK}, OpenKS={self.openK}, CloseKS={self.closeK}\n"
                f"in {self.cfg_path}"
            )
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Config nicht speichern:\n{e}")

    def export_edge(self):
        edge = self._compute_edge()
        arr = (edge.detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L")
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png")],
                                         initialfile="edge_map.png")
        if not p:
            return
        try:
            pil.save(p)
            messagebox.showinfo("Exportiert", f"Kantenkarte gespeichert:\n{p}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Kantenkarte nicht speichern:\n{e}")

# ------------------------------- Main -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.json", help="Pfad zur config.json")
    parser.add_argument("--img", type=str, default="", help="Pfad zum Bild")
    args = parser.parse_args()

    root = tk.Tk()
    app = EdgeTunerApp(root, args.config, args.img)
    root.mainloop()

if __name__ == "__main__":
    main()
