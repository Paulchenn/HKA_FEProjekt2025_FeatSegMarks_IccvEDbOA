import argparse, torch, numpy as np
import torch.nn.functional as F

# ========== 1) HILFSFUNKTIONEN ==========


def build_and_load(ckpt_path: str, device: str = "cuda"):
    print(f"[DryRun] build_and_load aufgerufen mit {ckpt_path}")
    class Dummy(torch.nn.Module):
        def forward(self, x): return torch.zeros((x.size(0), 10), device=x.device)
    return Dummy().to(device).eval()

def make_loaders():
    print("[DryRun] make_loaders aufgerufen")
    dummy_loader = [ (torch.randn(2,3,32,32), torch.tensor([0,1])) ]
    conflict_loader = [ (torch.randn(2,3,32,32), torch.tensor([0,1]), torch.tensor([0,1])) ]
    pair_loader = [ (torch.randn(2,3,32,32), torch.randn(2,3,32,32), torch.tensor([0,1])) ]
    return dummy_loader, conflict_loader, pair_loader

def feat_hook(model, x):
    return torch.randn(x.size(0), 128)  # Dummy-Feature


# ========== 2) METRIKEN (Paper) ==========

@torch.no_grad()
def clean_acc(model, loader, device="cuda"):
    correct, total = 0, 0
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return 100.0 * correct / max(1, total)

def acc_under_attack(model, loader, attack, device="cuda"):
    correct, total = 0, 0
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y) if attack.__class__.__name__ != "DeepFool" else attack(x)  # DeepFool ohne y
        pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return 100.0 * correct / max(1, total)

@torch.no_grad()
def sbGE(model, conflict_loader, device="cuda"):
    """ Anteil Vorhersagen nach Shape-Label auf Konfliktbildern """
    shape_correct, total = 0, 0
    model.eval()
    for x, y_shape, _y_tex in conflict_loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu()
        shape_correct += (pred == y_shape).sum().item()
        total += x.size(0)
    return 100.0 * shape_correct / max(1, total)

@torch.no_grad()
def sbIS(model, pair_loader, device="cuda"):
    """
    mittlere Korrelation der Feature-Vektoren für Paare (gleiche Form, andere Textur)
    """
    cors, n = 0.0, 0
    model.eval()
    for xa, xb, _shape in pair_loader:
        xa, xb = xa.to(device), xb.to(device)
        fa = feat_hook(model, xa)  # (B, F)
        fb = feat_hook(model, xb)
        fa = fa.flatten(1).cpu().numpy()
        fb = fb.flatten(1).cpu().numpy()
        for i in range(fa.shape[0]):
            aa = (fa[i] - fa[i].mean()) / (fa[i].std() + 1e-6)
            bb = (fb[i] - fb[i].mean()) / (fb[i].std() + 1e-6)
            cors += float((aa * bb).mean())  # Pearson approx
            n += 1
    return 100.0 * cors / max(1, n)

@torch.no_grad()
def eval_backdoor(model, clean_loader, add_trigger_fn, target_class: int, device="cuda"):
    """
    Clean-Acc + ASR (Attack Success Rate) für Backdoor.
    add_trigger_fn: x -> x' fügt Trigger ein (Pixel/Pattern).
    """
    # Clean
    correct, total = 0, 0
    for x, y in clean_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    clean_acc_val = 100.0 * correct / max(1, total)

    # ASR
    succ, total = 0, 0
    for x, _ in clean_loader:
        x = x.to(device)
        xt = add_trigger_fn(x)  # gleiche Größe, mit Trigger
        pred = model(xt).argmax(1).cpu()
        succ += (pred == target_class).sum().item()
        total += x.size(0)
    asr = 100.0 * succ / max(1, total)
    return clean_acc_val, asr


# ========== 3) KLEINER BACKDOOR-TRIGGER (falls du keinen hast) ==========

def add_pixel_trigger(x, intensity=0.5, size=3):
    """
    Einfacher weißer Block unten rechts als Trigger (für Demo).
    x: Tensor [B,C,H,W] in [-1,1] oder [0,1] – wir clippen vorsichtig.
    """
    B, C, H, W = x.shape
    xt = x.clone()
    s = size
    xt[..., H - s:H, W - s:W] = xt[..., H - s:H, W - s:W] * 0 + intensity
    return xt.clamp(x.min().item(), x.max().item())


def make_attacks(model, eps=8/255):
    # Dry-Run: keine Angriffe – leeres Dict vermeidet NameError
    return {}


# ========== 4) HAUPTPROGRAMM ==========

def main():
    ap = argparse.ArgumentParser("All-in-one Evaluation nach ICCV-Paper")
    ap.add_argument("--ckpt_baseline", required=True)
    ap.add_argument("--ckpt_sdboa",    required=True)
    ap.add_argument("--eps", type=float, default=8/255)   # 25/255 für FMNIST
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target_class", type=int, default=0)  # Backdoor-Zielklasse (anpassen)
    args = ap.parse_args()

    device = args.device

    # Modelle + Loader
    model_base  = build_and_load(args.ckpt_baseline, device)
    model_sdboa = build_and_load(args.ckpt_sdboa,    device)
    test_loader, conflict_loader, pair_loader = make_loaders()

    # Clean
    base_clean = clean_acc(model_base,  test_loader, device)
    sdb_clean  = clean_acc(model_sdboa, test_loader, device)

    # Adversarial
    base_atks = make_attacks(model_base,  eps=args.eps)
    sdb_atks  = make_attacks(model_sdboa, eps=args.eps)
    base_adv = {n: acc_under_attack(model_base,  test_loader, atk, device) for n, atk in base_atks.items()}
    sdb_adv  = {n: acc_under_attack(model_sdboa, test_loader, atk, device) for n, atk in sdb_atks.items()}

    # Shape-Bias
    base_sbge = sbGE(model_base,  conflict_loader, device)
    sdb_sbge  = sbGE(model_sdboa, conflict_loader, device)
    base_sbis = sbIS(model_base,  pair_loader, device)
    sdb_sbis  = sbIS(model_sdboa, pair_loader, device)

    # Backdoor (einfacher Demo-Trigger; ersetze durch deinen Trigger-Loader, wenn vorhanden)
    base_bclean, base_asr = eval_backdoor(model_base,  test_loader, add_pixel_trigger, args.target_class, device)
    sdb_bclean,  sdb_asr  = eval_backdoor(model_sdboa, test_loader, add_pixel_trigger, args.target_class, device)

    # Ausgabe im Paper-Stil
    print("\n=== Evaluation nach ICCV (SDbOA) – Alles in einem Skript ===")
    print(f"Clean Acc            | base={base_clean:.2f} | sdboa={sdb_clean:.2f}")
    print(f"sbGE / sbIS          | base={base_sbge:.2f}/{base_sbis:.2f} | sdboa={sdb_sbge:.2f}/{sdb_sbis:.2f}")
    for n in sorted(set(list(base_adv.keys()) + list(sdb_adv.keys()))):
        b = base_adv.get(n, float('nan')); s = sdb_adv.get(n, float('nan'))
        print(f"{n:<10}          | base={b:.2f} | sdboa={s:.2f}")
    print(f"Backdoor Clean / ASR | base={base_bclean:.2f} / {base_asr:.2f} | sdboa={sdb_bclean:.2f} / {sdb_asr:.2f}")
    print("\nHinweis: Für identische Bedingungen auf gleiche Epochen/Batch/LR achten. "
          "Für ImageNet/CIFAR-10 eps=8/255; für FMNIST 25/255.")

if __name__ == "__main__":
    main()
