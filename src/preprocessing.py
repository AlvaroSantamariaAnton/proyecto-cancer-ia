"""
src/preprocessing.py
====================
Pipeline de preprocesado: selección de features, codificación, split estratificado
en train/val/test, escalado y cálculo de class weights.

Aplica las decisiones tomadas en el EDA (notebook 01_eda.ipynb):
- 21 features finales (7 bioquímicas + 7 genéticas + 4 clínicas + 2 hábitos + 1 edad)
- 'actividad_fisica' codificada como ordinal (Baja=0, Moderada=1, Alta=2)
- Variables de leakage, constantes y sin señal predictiva descartadas

Uso:
    from src.preprocessing import prepare_data
    data = prepare_data(df_master)
    # data contiene: X_train, X_val, X_test, y_train, y_val, y_test,
    #                scaler, class_weights, feature_names
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# === Constantes del pipeline ===

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TARGET = "cancer"

FEATURES_BIOQUIMICAS = ["glucosa", "colesterol", "trigliceridos", "hemoglobina",
                        "leucocitos", "plaquetas", "creatinina"]
FEATURES_GENETICAS = ["mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
                      "mut_PIK3CA", "mut_ALK", "mut_BRAF"]
FEATURES_CLINICAS = ["diabetes", "hipertension", "obesidad", "epoc"]
FEATURES_HABITOS = ["fumador", "actividad_fisica"]
FEATURES_DEMO = ["edad"]

FEATURES_FINALES = (FEATURES_BIOQUIMICAS + FEATURES_GENETICAS +
                    FEATURES_CLINICAS + FEATURES_HABITOS + FEATURES_DEMO)

# Codificación ordinal: preserva el orden Baja < Moderada < Alta
MAPEO_ACTIVIDAD = {"Baja": 0, "Moderada": 1, "Alta": 2}


# === Funciones del pipeline ===

def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Selecciona las 21 features finales del df_master y codifica las categóricas.

    Parameters
    ----------
    df : pd.DataFrame
        df_master con las 38 columnas originales (incluido paciente_id y target).

    Returns
    -------
    X : pd.DataFrame
        DataFrame con las 21 features (todas numéricas tras la codificación).
    y : pd.Series
        Serie con la variable objetivo (cancer, valores 0/1).
    """
    # Verificación de columnas requeridas
    missing = [c for c in FEATURES_FINALES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el df: {missing}")

    X = df[FEATURES_FINALES].copy()
    y = df[TARGET].copy()

    # Codificación ordinal de actividad_fisica.
    # Aplicamos siempre que NO sea ya numérica (más robusto que comprobar 'object').
    if not pd.api.types.is_numeric_dtype(X["actividad_fisica"]):
        X["actividad_fisica"] = X["actividad_fisica"].map(MAPEO_ACTIVIDAD)
        if X["actividad_fisica"].isnull().any():
            valores_extranos = df.loc[X["actividad_fisica"].isnull(),
                                       "actividad_fisica"].unique()
            raise ValueError(
                f"Hay valores en 'actividad_fisica' fuera del mapeo "
                f"{list(MAPEO_ACTIVIDAD.keys())}: {valores_extranos}"
            )
        # Forzamos a int (el .map deja float64 si hubo algún NaN intermedio)
        X["actividad_fisica"] = X["actividad_fisica"].astype("int64")

    # Validación final: todo debe ser numérico
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Columnas no numéricas tras la codificación: {non_numeric}")

    return X, y


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Split estratificado en tres particiones: train / val / test.

    Estrategia (estratificada en 2 pasos para mantener proporciones exactas):
      1. Separamos test (20% del total) del resto (80%).
      2. Del 80% restante, separamos val (25% del 80% = 20% del total).

    Resultado: 60% train, 20% val, 20% test.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
    test_size : float
        Proporción del total para test. Default 0.20.
    val_size : float
        Proporción del total para val. Default 0.20.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    dict con keys X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # Paso 1: separar test del resto
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Paso 2: del trainval (80% del total), separar val
    # val_size_relative ajusta para que val sea val_size del TOTAL, no del trainval
    val_size_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_relative,
        stratify=y_trainval,
        random_state=random_state,
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }


def scale_features(splits: dict) -> tuple[dict, StandardScaler]:
    """
    Aplica StandardScaler ajustado SOLO con X_train.
    Las particiones val y test reciben la transformación, no el ajuste.

    Esto evita data leakage: las medias y desviaciones que se aplican a val/test
    no contienen información de esas particiones.

    Devuelve splits con X_* convertidos a np.ndarray escalados, manteniendo y_* intactos.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(splits["X_train"])
    X_val_s   = scaler.transform(splits["X_val"])
    X_test_s  = scaler.transform(splits["X_test"])

    scaled = {
        "X_train": X_train_s, "y_train": splits["y_train"].values,
        "X_val":   X_val_s,   "y_val":   splits["y_val"].values,
        "X_test":  X_test_s,  "y_test":  splits["y_test"].values,
    }
    return scaled, scaler


def compute_class_weights_balanced(y_train: np.ndarray) -> dict:
    """
    Calcula los class weights balanceados para gestionar el desbalance.

    Fórmula: weight_class_i = n_samples / (n_classes * n_samples_in_class_i)
    Con prevalencia 19.3%, el resultado típico es:
      clase 0 (no cáncer):  ~0.62  (peso menor, hay muchos)
      clase 1 (cáncer):     ~2.59  (peso mayor, hay pocos)

    Returns
    -------
    dict[int, float]
        Diccionario {0: peso_0, 1: peso_1} listo para sklearn y Keras.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def prepare_data(df_master: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Pipeline completo de preprocesado.

    Parameters
    ----------
    df_master : pd.DataFrame
        Salida de data_loader.load_master_dataset().
    verbose : bool
        Si True, imprime un resumen del proceso.

    Returns
    -------
    dict con keys:
        X_train, X_val, X_test : np.ndarray escalados
        y_train, y_val, y_test : np.ndarray
        feature_names          : list[str]  (orden de las columnas)
        scaler                 : StandardScaler ya ajustado
        class_weights          : dict[int, float]
    """
    # 1. Selección de features y codificación
    X, y = select_features(df_master)

    # 2. Split train/val/test estratificado
    splits = split_train_val_test(X, y)

    # 3. Escalado (solo se ajusta con train)
    scaled, scaler = scale_features(splits)

    # 4. Class weights
    class_weights = compute_class_weights_balanced(scaled["y_train"])

    result = {
        **scaled,
        "feature_names": FEATURES_FINALES,
        "scaler": scaler,
        "class_weights": class_weights,
    }

    if verbose:
        print("=" * 60)
        print("  PIPELINE DE PREPROCESADO COMPLETADO")
        print("=" * 60)
        print(f"  Features seleccionadas: {len(FEATURES_FINALES)}")
        print(f"  Splits (train/val/test):")
        for split in ["train", "val", "test"]:
            X_s = result[f"X_{split}"]
            y_s = result[f"y_{split}"]
            n = len(y_s)
            n_pos = int(y_s.sum())
            prev = n_pos / n * 100
            print(f"    {split:5s} → {n:>6,} muestras  |  "
                  f"positivos: {n_pos:>5,} ({prev:.2f}%)")
        print(f"  Class weights: {class_weights}")
        print(f"  Escalado: media de X_train ≈ 0, std ≈ 1")
        print(f"    media columna 0: {result['X_train'][:, 0].mean():+.6f}")
        print(f"    std columna 0:   {result['X_train'][:, 0].std():+.6f}")

    return result


# === Persistencia ===

def save_splits(data: dict, out_dir: Path = PROCESSED_DIR) -> None:
    """
    Guarda los splits procesados en disco para reutilizar en las fases siguientes.

    Crea:
      - data/processed/X_train.parquet, X_val.parquet, X_test.parquet
      - data/processed/y_train.parquet, y_val.parquet, y_test.parquet
      - models/scaler.joblib
      - data/processed/preprocessing_meta.json (feature_names, class_weights, etc.)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = data["feature_names"]

    for split in ["train", "val", "test"]:
        X = pd.DataFrame(data[f"X_{split}"], columns=feature_names)
        y = pd.DataFrame({"cancer": data[f"y_{split}"]})
        X.to_parquet(out_dir / f"X_{split}.parquet", index=False)
        y.to_parquet(out_dir / f"y_{split}.parquet", index=False)

    joblib.dump(data["scaler"], MODELS_DIR / "scaler.joblib")

    meta = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "class_weights": data["class_weights"],
        "random_state": RANDOM_STATE,
        "shapes": {
            "X_train": list(data["X_train"].shape),
            "X_val":   list(data["X_val"].shape),
            "X_test":  list(data["X_test"].shape),
        },
    }
    with open(out_dir / "preprocessing_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  Splits guardados en: {out_dir}")
    print(f"  Scaler guardado en:  {MODELS_DIR / 'scaler.joblib'}")


def load_splits(in_dir: Path = PROCESSED_DIR) -> dict:
    """
    Carga los splits previamente guardados.

    Returns
    -------
    dict con la misma estructura que prepare_data() devuelve.
    """
    with open(in_dir / "preprocessing_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    class_weights = {int(k): float(v) for k, v in meta["class_weights"].items()}
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")

    data = {"feature_names": feature_names,
            "scaler": scaler,
            "class_weights": class_weights}

    for split in ["train", "val", "test"]:
        data[f"X_{split}"] = pd.read_parquet(in_dir / f"X_{split}.parquet").values
        data[f"y_{split}"] = pd.read_parquet(in_dir / f"y_{split}.parquet")["cancer"].values

    return data


if __name__ == "__main__":
    # Smoke test: ejecutar todo el pipeline desde cero
    from data_loader import load_master_dataset

    print("Cargando dataset...")
    df = load_master_dataset(verbose=False)

    print("\nProcesando...")
    data = prepare_data(df, verbose=True)

    print("\nGuardando en disco...")
    save_splits(data)

    print("\nReleyendo desde disco para verificar...")
    data_reloaded = load_splits()
    assert data_reloaded["X_train"].shape == data["X_train"].shape
    assert data_reloaded["class_weights"] == data["class_weights"]
    print("  Lectura verificada: las shapes y class_weights coinciden.")