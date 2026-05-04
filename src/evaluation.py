"""
src/evaluation.py
=================
Funciones reutilizables para evaluar modelos de clasificación binaria.

Calcula las 5 métricas exigidas por el enunciado:
  - Precisión, Recall, F1-Score (clase positiva = cancer = 1)
  - AUC-ROC
  - Accuracy (solo como referencia)

Incluye también:
  - Búsqueda del umbral óptimo de clasificación (F1 maximizado en validación)
  - Visualizaciones consistentes para las diapositivas finales
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score,
)

# Estilo visual consistente
PALETTE = {
    "neg": "#3498db",        # azul: clase negativa
    "pos": "#e74c3c",        # rojo: clase positiva
    "primary": "#2c3e50",    # azul oscuro para textos
    "accent": "#27ae60",     # verde para destacar el mejor
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


# === Cálculo de métricas ===

def evaluate_with_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Calcula las métricas binarias dadas las probabilidades y un umbral.

    Útil cuando ya tenemos las probabilidades calculadas (por ejemplo, tras
    optimizar el umbral en validación) y queremos evaluar con el umbral elegido.

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas reales (0/1).
    y_proba : np.ndarray
        Probabilidades predichas para la clase positiva.
    threshold : float
        Umbral de decisión. Default 0.5.

    Returns
    -------
    dict con métricas: precision, recall, f1, auc_roc, accuracy,
    average_precision, threshold, confusion_matrix.
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_true, y_proba)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": cm,
        "n_samples": int(len(y_true)),
        "n_positives": int(y_true.sum()),
    }


def evaluate_model_full(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Evalúa un modelo entrenado calculando todas las métricas obligatorias.

    Detecta automáticamente si el modelo tiene predict_proba (sklearn) o si
    devuelve probabilidades directamente con .predict() (Keras).

    Parameters
    ----------
    model : objeto entrenado
        Modelo con predict_proba (sklearn) o predict (Keras MLP).
    X : np.ndarray
        Features ya escaladas.
    y_true : np.ndarray
        Etiquetas reales.
    threshold : float
        Umbral de decisión. Default 0.5.

    Returns
    -------
    dict con métricas + array y_proba para reusar en visualizaciones.
    """
    if hasattr(model, "predict_proba"):
        # Modelos sklearn-like
        y_proba = model.predict_proba(X)[:, 1]
    else:
        # Keras: predict() devuelve directamente probabilidades para sigmoid
        y_proba = model.predict(X, verbose=0).ravel()

    metrics = evaluate_with_threshold(y_true, y_proba, threshold)
    metrics["y_proba"] = y_proba
    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray = None,
) -> dict:
    """
    Busca el umbral que maximiza una métrica dada en el rango [0.10, 0.90].

    Se debe usar SIEMPRE sobre el conjunto de VALIDACIÓN, nunca sobre test
    (eso sería data leakage, como advierte el PDF del enunciado).

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas del conjunto de validación.
    y_proba : np.ndarray
        Probabilidades predichas en validación.
    metric : str
        Métrica a maximizar. Default "f1".
    thresholds : np.ndarray
        Umbrales a probar. Default: 0.10 a 0.90, paso 0.01 (81 valores).

    Returns
    -------
    dict con: best_threshold, best_score, all_results (DataFrame).
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    rows = []
    for t in thresholds:
        m = evaluate_with_threshold(y_true, y_proba, threshold=t)
        rows.append({
            "threshold": t,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
        })

    df_results = pd.DataFrame(rows)
    best_idx = df_results[metric].idxmax()
    best_row = df_results.loc[best_idx]

    return {
        "best_threshold": float(best_row["threshold"]),
        "best_score": float(best_row[metric]),
        "metric": metric,
        "all_results": df_results,
    }


# === Comparativa de modelos ===

def compare_models(results: dict) -> pd.DataFrame:
    """
    Construye una tabla comparativa de varios modelos.

    Parameters
    ----------
    results : dict
        Diccionario {nombre_modelo: dict_métricas}.

    Returns
    -------
    pd.DataFrame ordenado por F1 descendente.
    """
    rows = []
    for name, m in results.items():
        rows.append({
            "Modelo": name,
            "Precisión": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score": round(m["f1"], 4),
            "AUC-ROC": round(m["auc_roc"], 4),
            "Accuracy": round(m["accuracy"], 4),
            "Threshold": round(m["threshold"], 2),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("F1-Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1   # ranking empieza en 1
    return df


# === Visualizaciones ===

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Matriz de confusión",
    save_path: Path = None,
    ax=None,
) -> None:
    """
    Pinta una matriz de confusión 2x2 con conteos absolutos y porcentajes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Construimos las anotaciones combinando conteos absolutos y % por fila
    total = cm.sum()
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"
                       for j in range(2)] for i in range(2)])

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=["Pred. No cáncer", "Pred. Cáncer"],
                yticklabels=["Real No cáncer", "Real Cáncer"],
                cbar=False, ax=ax, linewidths=1, linecolor="white",
                annot_kws={"size": 11, "fontweight": "bold"})
    ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Información adicional debajo
    tn, fp, fn, tp = cm.ravel()
    info = (f"VP={tp:,}  FP={fp:,}  FN={fn:,}  VN={tn:,}\n"
            f"Total: {total:,}")
    ax.text(0.5, -0.15, info, ha="center", transform=ax.transAxes,
            fontsize=9, color="gray")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")


def plot_roc_curves(
    results: dict,
    y_true: np.ndarray,
    title: str = "Curvas ROC comparativas",
    save_path: Path = None,
) -> None:
    """
    Pinta las curvas ROC de varios modelos sobre los mismos ejes.

    Parameters
    ----------
    results : dict
        {nombre_modelo: dict_métricas con clave 'y_proba'}.
    y_true : np.ndarray
        Etiquetas reales (las mismas para todos los modelos).
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Diagonal de referencia (clasificador aleatorio)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            linewidth=1, label="Aleatorio (AUC=0.5)")

    # Una curva por modelo
    for name, m in results.items():
        fpr, tpr, _ = roc_curve(y_true, m["y_proba"])
        auc = m["auc_roc"]
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.4f})")

    ax.set_xlabel("Tasa de falsos positivos (1 - Especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (Recall)")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")


def plot_pr_curves(
    results: dict,
    y_true: np.ndarray,
    title: str = "Curvas Precisión-Recall comparativas",
    save_path: Path = None,
) -> None:
    """
    Pinta las curvas Precisión-Recall de varios modelos.

    En problemas desbalanceados como este, PR es más informativa que ROC.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Línea base: prevalencia (precisión de un clasificador aleatorio)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline (prevalencia={baseline:.3f})")

    for name, m in results.items():
        precision, recall, _ = precision_recall_curve(y_true, m["y_proba"])
        ap = m["average_precision"]
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")


def plot_metrics_comparison(
    df_compare: pd.DataFrame,
    title: str = "Comparativa de métricas entre modelos",
    save_path: Path = None,
) -> None:
    """
    Gráfico de barras agrupadas comparando las 4 métricas principales por modelo.
    """
    metric_cols = ["Precisión", "Recall", "F1-Score", "AUC-ROC"]
    n_metrics = len(metric_cols)
    n_models = len(df_compare)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.8), 6))
    x = np.arange(n_models)
    width = 0.18

    colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]

    for i, metric in enumerate(metric_cols):
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, df_compare[metric], width,
                      label=metric, color=colors[i], edgecolor="black")
        for bar, v in zip(bars, df_compare[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(df_compare["Modelo"], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", ncol=4)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")


# === Smoke test ===

if __name__ == "__main__":
    # Verificación rápida con datos simulados
    print("Probando evaluation.py con datos simulados...\n")
    np.random.seed(42)
    n = 1000
    y_true = (np.random.rand(n) < 0.2).astype(int)   # ~20% positivos
    y_proba = np.clip(np.random.rand(n) + y_true * 0.3, 0, 1)

    metrics = evaluate_with_threshold(y_true, y_proba, threshold=0.5)
    print(f"Métricas con threshold=0.5:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  confusion_matrix:\n{metrics['confusion_matrix']}")

    print("\nBuscando umbral óptimo...")
    best = find_best_threshold(y_true, y_proba, metric="f1")
    print(f"  Mejor threshold: {best['best_threshold']:.2f}")
    print(f"  Mejor F1:        {best['best_score']:.4f}")

    print("\nOK — evaluation.py funciona correctamente.")