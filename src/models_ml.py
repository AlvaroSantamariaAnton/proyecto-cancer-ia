"""
src/models_ml.py
================
Definición de los 4 modelos clásicos de Machine Learning para el caso.

Modelos incluidos:
  1. Regresión Logística (baseline)
  2. Random Forest
  3. XGBoost
  4. LightGBM

Todos gestionan el desbalance de clases mediante class_weight='balanced'
(o el equivalente scale_pos_weight para los gradient boosting).
"""
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42


def get_logistic_regression() -> LogisticRegression:
    """
    Regresión Logística con regularización L2 y class_weight balanceado.

    Sirve de baseline lineal. Si los modelos complejos no la superan
    claramente, hay un problema en los datos o en el preprocesado.

    Notas (sklearn 1.8):
    - 'penalty' está deprecada. La regularización L2 es ahora el default
      cuando l1_ratio=0 (también el default), así que NO especificamos penalty.
    - 'n_jobs' fue deprecada para LogisticRegression: scikit-learn lo ignora.
    """
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def get_random_forest() -> RandomForestClassifier:
    """
    Random Forest con limitación de profundidad para evitar overfit.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def get_xgboost(scale_pos_weight: float = 4.18) -> XGBClassifier:
    """
    XGBoost con configuración estándar para clasificación binaria desbalanceada.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio n_negativos/n_positivos del set de entrenamiento.
        Se calcula automáticamente y se pasa al instanciar.
    """
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_lightgbm(scale_pos_weight: float = 4.18) -> LGBMClassifier:
    """
    LightGBM con configuración análoga a XGBoost para tener una comparativa robusta.
    """
    return LGBMClassifier(
        n_estimators=400,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def get_all_ml_models(scale_pos_weight: float = 4.18) -> dict:
    """
    Devuelve los 4 modelos clásicos instanciados, listos para entrenar.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio n_neg/n_pos del set de entrenamiento.

    Returns
    -------
    dict[str, modelo] con los 4 modelos.
    """
    return {
        "Logistic Regression": get_logistic_regression(),
        "Random Forest":       get_random_forest(),
        "XGBoost":             get_xgboost(scale_pos_weight=scale_pos_weight),
        "LightGBM":            get_lightgbm(scale_pos_weight=scale_pos_weight),
    }


if __name__ == "__main__":
    print("Probando models_ml.py...\n")
    models = get_all_ml_models(scale_pos_weight=4.18)
    for name, m in models.items():
        print(f"  {name}:")
        print(f"    {type(m).__name__}")
        print(f"    Primeros parámetros: {list(m.get_params().items())[:3]}")
        print()
    print("OK — models_ml.py funciona correctamente.")