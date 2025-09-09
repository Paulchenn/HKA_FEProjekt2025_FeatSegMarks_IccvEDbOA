#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import uuid
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Dateien 'Stage[12]_Epoch_###.png' um eine Nummer erhöhen (Standard: +76)."
    )
    parser.add_argument("folder", help="Pfad zum Ordner mit den Dateien")
    parser.add_argument("--offset", type=int, default=76, help="Offset, der addiert wird (Standard: 76)")
    parser.add_argument("--dry-run", action="store_true", help="Nur anzeigen, was passieren würde")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Fehler: Ordner nicht gefunden: {folder}")
        sys.exit(1)

    # Unterstützt Stage1_... und Stage2_..., Dateiendung case-insensitive
    pattern = re.compile(
        r"^(?P<prefix>Stage(?P<stage>[12])_Epoch_)(?P<num>\d+)(?P<ext>\.png)$",
        re.IGNORECASE
    )

    sources = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue

        num_str = m.group("num")
        old_width = len(num_str)
        new_num = int(num_str) + args.offset
        new_num_str = str(new_num).zfill(max(old_width, len(str(new_num))))
        # Präfix & Extension aus dem Original übernehmen (erhält Groß/Kleinschreibung)
        new_name = f"{m.group('prefix')}{new_num_str}{m.group('ext')}"
        target = p.with_name(new_name)
        sources.append((p, target))

    if not sources:
        print("Keine passenden Dateien gefunden (erwartet: Stage1/Stage2_Epoch_###.png).")
        return

    # Ziel-Kollisionen prüfen (nur blockierend, wenn Ziel nicht selbst Quelle ist)
    source_set = {s for s, _ in sources}
    target_set = {t for _, t in sources}
    blocking = [t for t in target_set if t.exists() and t not in source_set]
    if blocking:
        print("Abbruch: Folgende Ziel-Dateien existieren bereits und werden nicht mit umbenannt:")
        for t in sorted(blocking):
            print("  -", t.name)
        print("Bitte verschieben/entfernen oder anderen Ordner wählen.")
        sys.exit(2)

    # Anzeige der geplanten Änderungen
    print("Geplante Umbenennungen:")
    for s, t in sorted(sources, key=lambda x: x[0].name):
        print(f"  {s.name}  ->  {t.name}")

    if args.dry_run:
        print("\n(DRY-RUN) Es wurden keine Dateien geändert.")
        return

    # Zweiphasiges, kollisionssicheres Umbenennen
    temp_pairs = []
    for s, t in sources:
        tmp = s.with_name(f"__tmp__{uuid.uuid4().hex}__{s.name}")
        s.rename(tmp)
        temp_pairs.append((tmp, t))

    for tmp, t in temp_pairs:
        tmp.rename(t)

    print(f"Fertig: {len(sources)} Dateien umbenannt.")

if __name__ == "__main__":
    main()
