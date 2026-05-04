"""
src/data_loader.py
==================
Carga y unión de los 6 CSV del dataset de cáncer.

Uso:
    from src.data_loader import load_master_dataset
    df = load_master_dataset()
"""
from pathlib import Path
import pandas as pd

# Ruta a los datos crudos (relativa a la raíz del proyecto)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Nombres canónicos de los 6 CSV
CSV_FILES = {
    "bioquimicos":      "CASOCANCER_01_BIOQUIMICOS.csv",
    "clinicos":         "CASOCANCER_02_CLINICOS.csv",
    "geneticos":        "CASOCANCER_03_GENETICOS.csv",
    "economicos":       "CASOCANCER_04_ECONOMICOS.csv",
    "habitos":          "CASOCANCER_05_GENERALES.csv",
    "sociodemograficos": "CASOCANCER_06_SOCIODEMOGRAFICOS.csv",
}


def load_csv(file_name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Carga un CSV del dataset gestionando dos particularidades del origen:
    - Encoding UTF-8 con BOM (utf-8-sig limpia el carácter invisible inicial).
    - El CSV 04_ECONOMICOS usa coma como separador decimal en lugar de punto.

    Parameters
    ----------
    file_name : str
        Nombre del fichero CSV (sin path).
    raw_dir : Path
        Carpeta donde buscarlo. Por defecto, data/raw del proyecto.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos del CSV.
    """
    path = raw_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el fichero: {path}")

    if "04_ECONOMICOS" in file_name:
        return pd.read_csv(path, encoding="utf-8-sig", decimal=",")
    return pd.read_csv(path, encoding="utf-8-sig")


def load_all_collections(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """
    Carga las 6 colecciones del dataset y las devuelve en un diccionario.

    Returns
    -------
    dict[str, pd.DataFrame]
        Claves: nombre lógico de la colección. Valores: DataFrame.
    """
    return {name: load_csv(file_name, raw_dir)
            for name, file_name in CSV_FILES.items()}


def merge_collections(collections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Une las 6 colecciones en un único DataFrame por la clave 'paciente_id'.

    Usa join 'inner': solo conserva pacientes presentes en TODAS las colecciones.
    En el dataset oficial, esto debe coincidir con todos los pacientes (50.001).

    Returns
    -------
    pd.DataFrame
        DataFrame unificado.
    """
    keys = list(collections.keys())
    df = collections[keys[0]].copy()
    for name in keys[1:]:
        df = df.merge(collections[name], on="paciente_id", how="inner")
    return df


def load_master_dataset(raw_dir: Path = RAW_DIR, verbose: bool = True) -> pd.DataFrame:
    """
    Pipeline completo: carga las 6 colecciones y devuelve el DataFrame unificado.

    Parameters
    ----------
    raw_dir : Path
        Carpeta de datos crudos. Por defecto, data/raw.
    verbose : bool
        Si True, imprime las dimensiones del resultado.

    Returns
    -------
    pd.DataFrame
        df_master con 50.001 filas × 38 columnas (incluye paciente_id y target).
    """
    collections = load_all_collections(raw_dir)
    df_master = merge_collections(collections)

    if verbose:
        print(f"  load_master_dataset → shape: {df_master.shape}")
        print(f"  Columnas: {df_master.shape[1]}, Filas: {df_master.shape[0]}")
        print(f"  Nulos totales: {df_master.isnull().sum().sum()}")
        print(f"  Duplicados en paciente_id: {df_master['paciente_id'].duplicated().sum()}")

    return df_master


if __name__ == "__main__":
    # Si ejecutas el módulo directamente con `python src/data_loader.py`,
    # hace una prueba rápida.
    print("Probando data_loader...")
    df = load_master_dataset(verbose=True)
    print(f"\n  Primeras 3 filas:")
    print(df.head(3))