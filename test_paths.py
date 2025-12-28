#!/usr/bin/env python3
from pathlib import Path

print("Current working directory:", Path.cwd())
print()

# Check from project root
notebooks = Path('notebooks')
print(f"Notebooks folder exists: {notebooks.exists()}")

generated = Path('generated')
print(f"Generated folder exists: {generated.exists()}")

if generated.exists():
    print(f"  - fraud_embeddings.pkl exists: {(generated / 'fraud_embeddings.pkl').exists()}")
    print(f"  - fraud_data_combined_clean.csv exists: {(generated / 'fraud_data_combined_clean.csv').exists()}")
    print(f"  Files in generated: {list(generated.glob('*.pkl')) + list(generated.glob('*.csv'))}")

# Simulate what app.py would do
print("\n--- Simulating app.py path resolution ---")
notebooks_dir = Path("notebooks/app.py").parent.resolve()
print(f"notebooks_dir: {notebooks_dir}")
project_root = notebooks_dir.parent.resolve()
print(f"project_root: {project_root}")
generated_dir = project_root / 'generated'
print(f"generated_dir: {generated_dir}")
print(f"generated_dir exists: {generated_dir.exists()}")

if generated_dir.exists():
    embeddings = generated_dir / 'fraud_embeddings.pkl'
    print(f"  - embeddings file: {embeddings}")
    print(f"  - embeddings exists: {embeddings.exists()}")
