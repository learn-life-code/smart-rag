#!/usr/bin/env python3
"""Generate a public sample corpus + labels for the benchmark (no real data).

  python scripts/make_example.py
  python scripts/benchmark.py example_spec.xlsx --labels example_labels.csv \
         --reject "price of bitcoin;;how to bake a cake"
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Parts"
    ws.append(["id", "UFS_GB", "RAM_GB", "SXM", "PCB_PN", "Plant"])
    random.seed(1)
    rows = []
    for i in range(1, 201):
        row = [f"SKU{1000+i}", random.choice([64, 128, 256]),
               random.choice([8, 16, 24, 32]), random.choice(["yes", "no"]),
               f"PCB{2000+i}", random.choice(["Plant-A", "Plant-B"])]
        ws.append(row)
        rows.append(row)
    wb.save("example_spec.xlsx")

    # labels keyed off real generated values so the benchmark is honest
    def val(skuid, col):
        for r in rows:
            if r[0] == skuid:
                return r[col]
        return ""
    labels = [
        ("UFS_GB for SKU1042", str(val("SKU1042", 1))),
        ("RAM_GB for SKU1099", str(val("SKU1099", 2))),
        ("PCB_PN for SKU1150", "PCB2150"),
        ("Plant for SKU1007", str(val("SKU1007", 5))),
    ]
    with open("example_labels.csv", "w", newline="", encoding="utf-8") as f:
        for q, a in labels:
            f.write(f"{q},{a}\n")
    print("Wrote example_spec.xlsx (200 parts) + example_labels.csv")
    print("Run:  python scripts/benchmark.py example_spec.xlsx --labels "
          'example_labels.csv --reject "price of bitcoin;;how to bake a cake"')


if __name__ == "__main__":
    main()
