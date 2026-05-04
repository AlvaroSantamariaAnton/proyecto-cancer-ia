"""
Script de verificación del entorno.
Comprueba que las librerías clave funcionan y que se pueden leer los CSV.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf

# Versiones
print("=" * 60)
print("VERIFICACIÓN DEL ENTORNO")
print("=" * 60)
print(f"pandas       : {pd.__version__}")
print(f"numpy        : {np.__version__}")
print(f"scikit-learn : {sklearn.__version__}")
print(f"tensorflow   : {tf.__version__}")
print()

# Lectura de los CSV
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
print(f"Carpeta de datos: {RAW_DIR}")
print(f"¿Existe?         : {RAW_DIR.exists()}")
print()

csv_files = sorted(RAW_DIR.glob("CASOCANCER_*.csv"))
print(f"CSV encontrados: {len(csv_files)}")
print("-" * 60)

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    print(f"{csv_path.name:45s} → {df.shape[0]:>6} filas × {df.shape[1]:>3} columnas")

print()
print("Si ves 6 CSV listados arriba con sus dimensiones, todo OK.")