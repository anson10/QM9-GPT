import os
import pandas as pd
from rdkit import Chem

XYZ_DIR = "data/raw/"
OUT_FILE = "data/processed/qm9_full.csv"

records = []

for file in os.listdir(XYZ_DIR):
    if not file.endswith(".xyz"):
        continue

    path = os.path.join(XYZ_DIR, file)

    with open(path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        continue

    meta = lines[1].strip().split()

    try:
        record = {
            "molecule_id": file.replace(".xyz", ""),
            "U0": float(meta[1]),
            "U": float(meta[2]),
            "H": float(meta[3]),
            "G": float(meta[4]),
            "Cv": float(meta[5]),
            "gap": float(meta[6]),
            "mu": float(meta[7]),
            "alpha": float(meta[8]),
            "smiles": meta[9],
        }

        # Validate SMILES using RDKit
        mol = Chem.MolFromSmiles(record["smiles"])
        if mol is None:
            continue

        record["num_atoms"] = mol.GetNumAtoms()
        record["num_bonds"] = mol.GetNumBonds()

        records.append(record)

    except Exception:
        continue

df = pd.DataFrame(records)

# Final cleaning
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)
print("Total molecules:", len(df))
print(df.head())
