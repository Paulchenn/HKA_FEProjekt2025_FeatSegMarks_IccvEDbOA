#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

def sniff_dialect(path, encoding="utf-8-sig", sample_bytes=65536):
    with open(path, "r", encoding=encoding, newline="") as f:
        sample = f.read(sample_bytes)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        except csv.Error:
            # Fallback: Komma
            class _Default(csv.Dialect):
                delimiter = ","
                quotechar = '"'
                escapechar = None
                doublequote = True
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL
                skipinitialspace = False
            dialect = _Default
        # Header vorhanden? (nur Info – wir schreiben so oder so mit Header)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = True
    return dialect, has_header

def choose_column(fieldnames, requested):
    # Case-insensitive Zuordnung
    lower_map = {name.lower(): name for name in fieldnames}
    candidates = [requested.lower(), "epoch", "epochs"]
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    raise SystemExit(
        f"Fehler: Spalte '{requested}' nicht gefunden. Verfügbare Spalten: {fieldnames}"
    )

def to_int(value, rownum):
    s = str(value).strip()
    if s == "":
        raise ValueError(f"Leerer Wert in Zeile {rownum}.")
    try:
        # erlaubt "1" oder "1.0"
        return int(float(s))
    except Exception as e:
        raise ValueError(f"Kann Wert '{s}' in Zeile {rownum} nicht als Integer lesen.") from e

def main():
    p = argparse.ArgumentParser(
        description="Erhöht Werte in einer CSV-Spalte (Standard: 'epoch') um ein Offset (Standard: +76)."
    )
    p.add_argument("csv_in", help="Pfad zur Eingabe-CSV")
    p.add_argument("-o", "--output", help="Pfad zur Ausgabe-CSV (Standard: *_shifted.csv)")
    p.add_argument("--offset", type=int, default=76, help="Offset, das addiert wird (Default: 76)")
    p.add_argument("--column", default="epoch", help="Spaltenname (case-insensitiv). Default: epoch")
    p.add_argument("--dry-run", action="store_true", help="Nur anzeigen, nichts schreiben")
    p.add_argument("--inplace", action="store_true", help="Datei sicher in-place überschreiben")
    p.add_argument("--encoding", default="utf-8-sig", help="Datei-Encoding (Default: utf-8-sig)")
    args = p.parse_args()

    src = Path(args.csv_in)
    if not src.is_file():
        print(f"Fehler: Datei nicht gefunden: {src}", file=sys.stderr)
        sys.exit(1)

    dialect, _ = sniff_dialect(src, encoding=args.encoding)

    # Zielpfad bestimmen
    if args.inplace:
        out_path = None  # wir schreiben in Temp und ersetzen anschließend
    else:
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = src.with_name(f"{src.stem}_shifted{src.suffix or '.csv'}")

    # Einlesen + Vorschau / Schreiben
    updated_rows = 0
    total_rows = 0

    with open(src, "r", encoding=args.encoding, newline="") as fin:
        reader = csv.DictReader(fin, dialect=dialect)
        if reader.fieldnames is None:
            print("Fehler: Konnte Header nicht lesen (CSV ohne Kopfzeile?).", file=sys.stderr)
            sys.exit(2)

        epoch_col = choose_column(reader.fieldnames, args.column)

        # Dry-Run: nur zeigen, was passiert, anhand der ersten 5 Zeilen
        if args.dry_run:
            print(f"Spalte erkannt: '{epoch_col}'. Offset: {args.offset}")
            print("Vorschau (erste bis zu 5 Zeilen):")
            for i, row in enumerate(reader, start=1):
                total_rows += 1
                try:
                    old = to_int(row[epoch_col], i)
                    new = old + args.offset
                    updated_rows += 1
                    if i <= 5:
                        print(f"  Zeile {i}: {old} -> {new}")
                except ValueError as e:
                    if i <= 5:
                        print(f"  Zeile {i}: SKIP ({e})")
                if i >= 5:
                    break
            print(f"\n(DRY-RUN) Total gelesene Zeilen (bis 5 gezeigt): {total_rows}, "
                  f"angepasst (in Vorschau): {updated_rows}")
            return

        # Schreiben (entweder Temp->Replace oder direkte Ausgabedatei)
        if args.inplace:
            tmpfile = NamedTemporaryFile("w", delete=False, encoding=args.encoding, newline="")
            fout_path = Path(tmpfile.name)
            tmp_close_needed = True
        else:
            fout_path = out_path
            tmp_close_needed = False

        with (tmpfile if args.inplace else open(fout_path, "w", encoding=args.encoding, newline="")) as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, dialect=dialect)
            writer.writeheader()
            for i, row in enumerate(reader, start=1):
                total_rows += 1
                try:
                    old = to_int(row[epoch_col], i)
                    row[epoch_col] = str(old + args.offset)
                    updated_rows += 1
                except ValueError:
                    # Wert bleibt unverändert, aber Zeile wird geschrieben
                    pass
                writer.writerow(row)

        if args.inplace:
            if tmp_close_needed:
                fout.close()
            # Backup anlegen
            backup = src.with_suffix(src.suffix + ".bak")
            try:
                if backup.exists():
                    backup.unlink()
                os.replace(src, backup)
                os.replace(fout_path, src)
                print(f"Fertig. {updated_rows}/{total_rows} Zeilen angepasst. Backup: {backup.name}")
            except Exception as e:
                # Falls etwas schiefgeht, Temp-Datei nicht verlieren
                print(f"Fehler beim Ersetzen: {e}. Temporäre Datei: {fout_path}", file=sys.stderr)
                sys.exit(3)
        else:
            print(f"Fertig. {updated_rows}/{total_rows} Zeilen angepasst -> {fout_path}")

if __name__ == "__main__":
    main()
