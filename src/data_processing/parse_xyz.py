import os
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

def parse_qm9_to_csv(xyz_dir, output_path):
    records = []
    
    # Get all .xyz files and sort them
    files = sorted([f for f in os.listdir(xyz_dir) if f.endswith('.xyz')])
    print(f"Found {len(files)} files. Starting extraction...")

    # Conversion factor: 1 Hartree = 27.2114 eV
    H_TO_EV = 27.2114

    for filename in tqdm(files):
        path = os.path.join(xyz_dir, filename)
        
        with open(path, 'r') as f:
            lines = f.readlines()
            
            # --- EXTRACT PROPERTIES (Line 2 / index 1) ---
            # Index 7: HOMO, Index 8: LUMO, Index 9: GAP
            properties = lines[1].split()
            
            # Correcting the index to 9 for HOMO-LUMO Gap
            try:
                gap_hartree = float(properties[9])
                
                # --- EXTRACT SMILES ---
                # In official QM9, SMILES is usually on the 2nd to last line.
                # We check the 2nd to last line first.
                smiles = lines[-2].split()[0]
                
                # Validation: Some versions have SMILES on -1 or -2. 
                # If RDKit fails, we try the last line as a fallback.
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    smiles = lines[-1].split()[0]
                    mol = Chem.MolFromSmiles(smiles)

                if mol is not None:
                    records.append({
                        'molecule_id': filename.split('_')[-1].replace('.xyz', ''),
                        'smiles': smiles,
                        'gap_ev': gap_hartree * H_TO_EV, 
                    })
            except (IndexError, ValueError):
                continue

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df)} valid molecules to {output_path}")

if __name__ == "__main__":
    RAW_DIR = "data/raw" # Ensure this points to the folder with 134k files
    OUT_DIR = "data/processed/qm9_clean.csv"
    
    os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)
    parse_qm9_to_csv(RAW_DIR, OUT_DIR)