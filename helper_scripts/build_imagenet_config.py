#!/usr/bin/env python3
import json, urllib.request, pathlib

# Ziel-Datei
OUT = pathlib.Path("imagenet_config.json")

# Offizielle Klassenliste aus torchvision:
# Format: {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
URL = "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/imagenet_class_index.json"

with urllib.request.urlopen(URL) as r:
    class_index = json.load(r)

# Nur die Klarnamen (zweites Element) übernehmen
selected = {str(k): v[1] for k, v in sorted(((int(i), pair) for i, pair in class_index.items()))}

config = {
    "dataset": {
        "name": "ImageNet256",
        "path": "~/torch/data/ImageNet256",
        "SELECTED_SYNSETS": selected
    }
}

OUT.write_text(json.dumps(config, ensure_ascii=False, indent=2))
print(f"Geschrieben: {OUT.resolve()}")
