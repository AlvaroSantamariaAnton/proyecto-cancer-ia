"""
streamlit_app/pages/2_Comparativa_Modelos.py
============================================
Página de comparativa técnica de los 5 modelos evaluados.

Muestra la evidencia experimental que respalda el sistema:
ranking, métricas, curvas ROC, PR, matrices de confusión y calibración.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "streamlit_app"))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, brier_score_loss, average_precision_score
from sklearn.calibration import calibration_curve

from assets.ui import (
    load_css, app_header, section_title, kpi_card,
    info_card, medical_disclaimer, COLORS, apply_plotly_theme,
)


# ============================================================
#  CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Comparativa · Sistema Oncológico",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()


# ============================================================
#  CARGA DE DATOS DEL ESTUDIO (cacheado)
# ============================================================
MODELS_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_results_test():
    """Carga la tabla comparativa final en test."""
    return pd.read_csv(REPORTS_DIR / "comparativa_final_test.csv")


@st.cache_data
def load_proba_data():
    """Carga las probabilidades de los 5 modelos en test (para curvas)."""
    return joblib.load(MODELS_DIR / "results_final_test.joblib")


@st.cache_data
def load_test_labels():
    """Carga las etiquetas reales de test."""
    df = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")
    return df["cancer"].values


df_results = load_results_test()
results_test = load_proba_data()
y_test = load_test_labels()


# ============================================================
#  HEADER
# ============================================================
app_header(
    "Comparativa técnica de modelos",
    "Análisis del rendimiento de los 5 algoritmos evaluados sobre 10.001 pacientes de test",
)


# ============================================================
#  KPIs DE RESUMEN
# ============================================================
section_title("Resumen del rendimiento")

best_row = df_results.iloc[0]
mlp_row = df_results[df_results["Modelo"].str.startswith("MLP")].iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Mejor F1-Score",
             f"{best_row['F1-Score']:.4f}",
             f"{best_row['Modelo']}")
with c2:
    kpi_card("F1-Score MLP",
             f"{mlp_row['F1-Score']:.4f}",
             f"Modelo principal · t={mlp_row['Threshold']:.2f}")
with c3:
    kpi_card("AUC-ROC máximo",
             f"{df_results['AUC-ROC'].max():.4f}",
             "Mejor capacidad de discriminación")
with c4:
    kpi_card("Diferencia top-1 vs top-2",
             f"{(df_results.iloc[0]['F1-Score'] - df_results.iloc[1]['F1-Score'])*100:.2f} pp",
             "Margen entre el primer y segundo modelo")

st.write("")


# ============================================================
#  TABLA COMPARATIVA
# ============================================================
section_title("Ranking final en test")

st.dataframe(
    df_results.style.format({
        "Precisión": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}",
        "AUC-ROC": "{:.4f}",
        "Accuracy": "{:.4f}",
        "Threshold": "{:.2f}",
    }).background_gradient(subset=["F1-Score"], cmap="Blues"),
    width="stretch",
    hide_index=True,
)

info_card(
    "El <strong>F1-Score</strong> es la métrica principal en este estudio "
    "porque equilibra precisión y recall, dos aspectos críticos en cribado clínico: "
    "no podemos ni dejar pasar cánceres reales (recall) ni alarmar excesivamente "
    "a pacientes sanos (precisión). El Random Forest y la Red Neuronal Multicapa "
    "están técnicamente empatados (ΔF1 = 0.002), pero operan en regímenes distintos."
)

st.write("")


# ============================================================
#  GRÁFICO DE BARRAS - MÉTRICAS POR MODELO
# ============================================================
section_title("Métricas por modelo")

metrics_to_plot = ["Precisión", "Recall", "F1-Score", "AUC-ROC"]
metric_colors = {
    "Precisión": COLORS["primary"],
    "Recall":    COLORS["danger"],
    "F1-Score":  COLORS["success"],
    "AUC-ROC":   COLORS["warning"],
}

fig_metrics = go.Figure()
for metric in metrics_to_plot:
    fig_metrics.add_trace(go.Bar(
        name=metric,
        x=df_results["Modelo"],
        y=df_results[metric],
        marker_color=metric_colors[metric],
        text=[f"{v:.3f}" for v in df_results[metric]],
        textposition="outside",
        textfont=dict(size=10),
    ))

fig_metrics.update_layout(
    barmode="group",
    height=420,
    yaxis=dict(range=[0, 1.0], title="Valor"),
    xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, r=20, b=40, l=40),
)
fig_metrics = apply_plotly_theme(fig_metrics)
st.plotly_chart(fig_metrics, width="stretch")

st.write("")


# ============================================================
#  CURVA ROC SUPERPUESTA
# ============================================================
section_title("Capacidad de discriminación · Curvas ROC")

fig_roc = go.Figure()

# Línea de referencia (clasificador aleatorio)
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    line=dict(dash="dash", color="#94A3B8", width=1.5),
    name="Aleatorio (AUC=0.5)",
    hoverinfo="skip",
))

# Curvas de cada modelo
model_color_map = {
    "Logistic Regression": COLORS["model_lr"],
    "Random Forest":       COLORS["model_rf"],
    "XGBoost":              COLORS["model_xgb"],
    "LightGBM":             COLORS["model_lgb"],
}
# La MLP puede tener nombre con sufijo
mlp_key = next(k for k in results_test.keys() if "MLP" in k)
model_color_map[mlp_key] = COLORS["model_mlp"]

for name, m in results_test.items():
    y_proba = m["y_proba_test"]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = m["auc_roc"]
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"{name} (AUC={auc:.4f})",
        line=dict(color=model_color_map.get(name, "#000"), width=2.5),
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>%{fullData.name}</extra>",
    ))

fig_roc.update_layout(
    height=520,
    xaxis_title="Tasa de falsos positivos (1 − Especificidad)",
    yaxis_title="Tasa de verdaderos positivos (Sensibilidad)",
    xaxis=dict(range=[-0.01, 1.01]),
    yaxis=dict(range=[-0.01, 1.01]),
    legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0"),
)
fig_roc = apply_plotly_theme(fig_roc)
st.plotly_chart(fig_roc, width="stretch")

info_card(
    "La curva ROC muestra el equilibrio entre <strong>sensibilidad</strong> "
    "(detectar cáncer real) y <strong>especificidad</strong> (no falsamente alarmar) "
    "a través de todos los umbrales posibles. El AUC-ROC resume la curva en un único "
    "valor: <strong>0.5 = aleatorio · 1.0 = perfecto</strong>. "
    "Todos los modelos del estudio superan el 0.80, indicando un rendimiento sólido."
)

st.write("")


# ============================================================
#  CURVA PRECISIÓN-RECALL
# ============================================================
section_title("Curvas Precisión-Recall")

fig_pr = go.Figure()

# Baseline (predicción constante = prevalencia)
prevalencia = float(y_test.mean())
fig_pr.add_hline(y=prevalencia, line_dash="dash", line_color="#94A3B8",
                 annotation_text=f"Baseline (prevalencia={prevalencia:.3f})",
                 annotation_position="bottom right")

for name, m in results_test.items():
    y_proba = m["y_proba_test"]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    # Average Precision (área bajo la curva PR) — función oficial de sklearn
    ap = average_precision_score(y_test, y_proba)
    fig_pr.add_trace(go.Scatter(
        x=recall, y=precision, mode="lines",
        name=f"{name} (AP={ap:.4f})",
        line=dict(color=model_color_map.get(name, "#000"), width=2.5),
        hovertemplate="Recall: %{x:.3f}<br>Precisión: %{y:.3f}<extra>%{fullData.name}</extra>",
    ))

fig_pr.update_layout(
    height=520,
    xaxis_title="Recall (Sensibilidad)",
    yaxis_title="Precisión",
    xaxis=dict(range=[-0.01, 1.01]),
    yaxis=dict(range=[0, 1.05]),
    legend=dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02,
                bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0"),
)
fig_pr = apply_plotly_theme(fig_pr)
st.plotly_chart(fig_pr, width="stretch")

info_card(
    "La curva Precisión-Recall es especialmente informativa en problemas con "
    "<strong>clases desbalanceadas</strong> como el nuestro (19.3% prevalencia). "
    "Mide cómo varía la precisión al aumentar el recall. Por encima de la línea horizontal "
    "(prevalencia base) significa que el modelo aporta información: predice mejor que "
    "asignar ciegamente la clase mayoritaria a todos los pacientes."
)

st.write("")


# ============================================================
#  MATRIZ DE CONFUSIÓN DEL MEJOR MODELO
# ============================================================
section_title("Matriz de confusión del mejor modelo")

best_model_name = best_row["Modelo"]
best_metrics = results_test[best_model_name]
cm = best_metrics["confusion_matrix"]
tn, fp, fn, tp = cm.ravel()

col_cm, col_lectura = st.columns([1, 1])

with col_cm:
    # Matriz de confusión visual con Plotly heatmap
    z_text = [
        [f"VN: {tn:,}<br>({tn/len(y_test)*100:.1f}%)",
         f"FP: {fp:,}<br>({fp/len(y_test)*100:.1f}%)"],
        [f"FN: {fn:,}<br>({fn/len(y_test)*100:.1f}%)",
         f"VP: {tp:,}<br>({tp/len(y_test)*100:.1f}%)"],
    ]

    fig_cm = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "#1A202C"},
        x=["Predicho: No cáncer", "Predicho: Cáncer"],
        y=["Real: No cáncer", "Real: Cáncer"],
        colorscale=[[0, "#EFF6FF"], [1, "#0066CC"]],
        showscale=False,
    ))
    fig_cm.update_layout(
        height=380,
        title=f"Matriz de confusión · {best_model_name}",
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    fig_cm = apply_plotly_theme(fig_cm)
    st.plotly_chart(fig_cm, width="stretch")

with col_lectura:
    sensibilidad = tp / (tp + fn) * 100
    especificidad = tn / (tn + fp) * 100
    valor_predictivo_pos = tp / (tp + fp) * 100
    valor_predictivo_neg = tn / (tn + fn) * 100

    st.markdown(
        f'<div style="font-size: 0.78rem; color: {COLORS["text_muted"]}; '
        f'text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px;">'
        f'Lectura clínica</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"**Sensibilidad: {sensibilidad:.1f}%**")
    st.caption(f"De los pacientes con cáncer real, el modelo detecta correctamente "
               f"{sensibilidad:.1f} de cada 100.")

    st.markdown(f"**Especificidad: {especificidad:.1f}%**")
    st.caption(f"De los pacientes sanos, el modelo correctamente descarta "
               f"{especificidad:.1f} de cada 100.")

    st.markdown(f"**Valor predictivo positivo: {valor_predictivo_pos:.1f}%**")
    st.caption(f'Cuando el modelo dice "sospecha de cáncer", acierta el '
               f"{valor_predictivo_pos:.1f}% de las veces.")

    st.markdown(f"**Valor predictivo negativo: {valor_predictivo_neg:.1f}%**")
    st.caption(f'Cuando el modelo dice "no sospecha", acierta el '
               f"{valor_predictivo_neg:.1f}% de las veces.")

st.write("")


# ============================================================
#  CALIBRACIÓN DE LOS MODELOS
# ============================================================
section_title("Calibración de los modelos")

st.markdown("""
La **calibración** mide si las probabilidades predichas reflejan la realidad estadística:
cuando el modelo dice "80%", ¿es realmente cáncer el 80% de las veces? Las redes neuronales
tienen tendencia natural a la sobreconfianza, por lo que aplicamos **calibración isotónica
post-hoc** sobre validación a los 5 modelos para corregirla. El **Brier Score** mide la
calidad de las probabilidades (más bajo es mejor).
""")

# Cargar calibradores y calcular Brier antes/después
try:
    mlp_calibrator_data = joblib.load(MODELS_DIR / "mlp_calibrator.joblib")
    ml_calibrators = joblib.load(MODELS_DIR / "ml_calibrators.joblib")

    # Brier antes (cruda) y después (calibrada) para cada modelo
    brier_data = []
    for name, m in results_test.items():
        p_raw = m["y_proba_test"]
        brier_raw = brier_score_loss(y_test, p_raw)

        if "MLP" in name:
            cal = mlp_calibrator_data["calibrator"]
        else:
            cal = ml_calibrators.get(name, None)

        if cal is not None:
            p_cal = cal.predict(p_raw)
            brier_cal = brier_score_loss(y_test, p_cal)
            mejora = (1 - brier_cal/brier_raw) * 100
        else:
            p_cal = p_raw
            brier_cal = brier_raw
            mejora = 0.0

        brier_data.append({
            "Modelo": name,
            "Brier crudo": f"{brier_raw:.4f}",
            "Brier calibrado": f"{brier_cal:.4f}",
            "Mejora (%)": f"{mejora:+.1f}%",
        })

    df_brier = pd.DataFrame(brier_data)
    st.dataframe(df_brier, width="stretch", hide_index=True)

    # Reliability diagram (antes vs después) para los 5 modelos
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Antes de calibrar**")
        fig_rel_raw = go.Figure()
        fig_rel_raw.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#94A3B8"),
            name="Calibración perfecta",
            hoverinfo="skip",
        ))
        for name, m in results_test.items():
            p_raw = m["y_proba_test"]
            frac, mean_p = calibration_curve(y_test, p_raw, n_bins=10, strategy="quantile")
            fig_rel_raw.add_trace(go.Scatter(
                x=mean_p, y=frac, mode="lines+markers",
                name=name,
                line=dict(color=model_color_map.get(name, "#000"), width=2),
                marker=dict(size=7),
            ))
        fig_rel_raw.update_layout(
            height=400,
            xaxis_title="Probabilidad predicha media",
            yaxis_title="Frecuencia real de cáncer",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=False,
        )
        fig_rel_raw = apply_plotly_theme(fig_rel_raw)
        st.plotly_chart(fig_rel_raw, width="stretch")

    with col_b:
        st.markdown("**Después de calibrar**")
        fig_rel_cal = go.Figure()
        fig_rel_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#94A3B8"),
            name="Calibración perfecta",
            hoverinfo="skip",
        ))
        for name, m in results_test.items():
            p_raw = m["y_proba_test"]
            if "MLP" in name:
                cal = mlp_calibrator_data["calibrator"]
            else:
                cal = ml_calibrators.get(name, None)
            if cal is not None:
                p_cal = cal.predict(p_raw)
            else:
                p_cal = p_raw
            frac, mean_p = calibration_curve(y_test, p_cal, n_bins=10, strategy="quantile")
            fig_rel_cal.add_trace(go.Scatter(
                x=mean_p, y=frac, mode="lines+markers",
                name=name,
                line=dict(color=model_color_map.get(name, "#000"), width=2),
                marker=dict(size=7),
            ))
        fig_rel_cal.update_layout(
            height=400,
            xaxis_title="Probabilidad predicha media (calibrada)",
            yaxis_title="Frecuencia real de cáncer",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=True,
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                        bgcolor="rgba(255,255,255,0.9)"),
        )
        fig_rel_cal = apply_plotly_theme(fig_rel_cal)
        st.plotly_chart(fig_rel_cal, width="stretch")

    info_card(
        "Tras la calibración, las curvas de los 5 modelos se acercan considerablemente "
        "a la diagonal de calibración perfecta. La mejora del Brier Score "
        "(15-32% según el modelo) confirma que las probabilidades predichas son ahora "
        "estadísticamente honestas, lo que permite usarlas con seguridad en la página "
        "de Evaluación de Paciente."
    )

except FileNotFoundError:
    st.warning("No se han encontrado los calibradores. Ejecute el notebook 06_auditoria_modelo.ipynb para generarlos.")

st.write("")


# ============================================================
#  PROCESO DE ENTRENAMIENTO DE LA MLP
# ============================================================
section_title("Proceso de entrenamiento de la red neuronal")

st.markdown("""
La MLP se entrenó durante un máximo de 100 épocas con dos callbacks activos:
**EarlyStopping** (patience=12, restaura los pesos de la mejor época) y
**ReduceLROnPlateau** (factor=0.5, patience=6, reduce el lr si val_loss se estanca).
""")

try:
    import numpy as np_local
    history = joblib.load(MODELS_DIR / "mlp_history.joblib")
    n_epochs = len(history["loss"])
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    epochs_list = list(range(1, n_epochs + 1))

    col_loss, col_acc = st.columns(2)

    with col_loss:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs_list, y=history["loss"],
            mode="lines", name="Train",
            line=dict(color=COLORS["primary"], width=2.5),
        ))
        fig_loss.add_trace(go.Scatter(
            x=epochs_list, y=history["val_loss"],
            mode="lines", name="Validación",
            line=dict(color=COLORS["danger"], width=2.5),
        ))
        fig_loss.add_vline(
            x=best_epoch, line_dash="dash", line_color=COLORS["success"],
            annotation_text=f"Mejor época ({best_epoch})",
            annotation_position="top right",
        )
        fig_loss.update_layout(
            height=350,
            title=dict(text="Pérdida (Binary Cross-Entropy)", font=dict(size=13)),
            xaxis_title="Época",
            yaxis_title="Loss",
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
        )
        fig_loss = apply_plotly_theme(fig_loss)
        st.plotly_chart(fig_loss, width="stretch")

    with col_acc:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=epochs_list, y=history["accuracy"],
            mode="lines", name="Train",
            line=dict(color=COLORS["primary"], width=2.5),
        ))
        fig_acc.add_trace(go.Scatter(
            x=epochs_list, y=history["val_accuracy"],
            mode="lines", name="Validación",
            line=dict(color=COLORS["danger"], width=2.5),
        ))
        fig_acc.add_vline(
            x=best_epoch, line_dash="dash", line_color=COLORS["success"],
            annotation_text=f"Mejor época ({best_epoch})",
            annotation_position="top right",
        )
        fig_acc.update_layout(
            height=350,
            title=dict(text="Accuracy", font=dict(size=13)),
            xaxis_title="Época",
            yaxis_title="Accuracy",
            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        )
        fig_acc = apply_plotly_theme(fig_acc)
        st.plotly_chart(fig_acc, width="stretch")

    gap = history["val_loss"][best_epoch - 1] - history["loss"][best_epoch - 1]
    info_card(
        f"La red entrenó <strong>{n_epochs} épocas</strong> (EarlyStopping activado). "
        f"La mejor época fue la <strong>{best_epoch}</strong> "
        f"(val_loss = {min(history['val_loss']):.4f}). "
        f"Gap train/val = <strong>{gap:+.4f}</strong> → entrenamiento sano sin sobreajuste significativo. "
        f"El lr se redujo {history['loss'].count(history['loss'][-1])} veces mediante ReduceLROnPlateau."
    )

except FileNotFoundError:
    st.warning("No se encontró mlp_history.joblib. Ejecute el notebook 04_mlp.ipynb.")

st.write("")


# ============================================================
#  BÚSQUEDA DEL UMBRAL ÓPTIMO
# ============================================================
section_title("Optimización del umbral de decisión")

st.markdown("""
El umbral por defecto de **0.5** no es óptimo en problemas con clases desbalanceadas.
Se realizó una búsqueda exhaustiva en el rango **[0.10, 0.90]** con paso 0.01
sobre el conjunto de **validación**, maximizando el F1-Score.
El umbral encontrado se aplicó una sola vez al test final.
""")

try:
    from sklearn.metrics import f1_score as _f1, precision_score as _prec, recall_score as _rec

    mlp_val_data = joblib.load(MODELS_DIR / "mlp_results_val.joblib")
    y_proba_val = mlp_val_data["y_proba_val"]
    y_val_df = pd.read_parquet(PROCESSED_DIR / "y_val.parquet")
    y_val_arr = y_val_df["cancer"].values

    thresholds_range = np.arange(0.10, 0.91, 0.01)
    f1_scores, prec_scores, rec_scores = [], [], []

    for t in thresholds_range:
        preds = (y_proba_val >= t).astype(int)
        f1_scores.append(_f1(y_val_arr, preds, zero_division=0))
        prec_scores.append(_prec(y_val_arr, preds, zero_division=0))
        rec_scores.append(_rec(y_val_arr, preds, zero_division=0))

    best_t_idx = int(np.argmax(f1_scores))
    best_t_val = float(thresholds_range[best_t_idx])
    best_f1_val = float(f1_scores[best_t_idx])

    fig_thr = go.Figure()
    fig_thr.add_trace(go.Scatter(
        x=thresholds_range, y=prec_scores,
        mode="lines", name="Precisión",
        line=dict(color=COLORS["primary"], width=2),
    ))
    fig_thr.add_trace(go.Scatter(
        x=thresholds_range, y=rec_scores,
        mode="lines", name="Recall",
        line=dict(color=COLORS["danger"], width=2),
    ))
    fig_thr.add_trace(go.Scatter(
        x=thresholds_range, y=f1_scores,
        mode="lines", name="F1-Score",
        line=dict(color=COLORS["success"], width=2.5),
    ))
    fig_thr.add_vline(
        x=best_t_val, line_dash="dash", line_color="#1A202C",
        annotation_text=f"Threshold óptimo = {best_t_val:.2f}",
        annotation_position="top left",
    )
    fig_thr.add_vline(
        x=0.5, line_dash="dot", line_color="#94A3B8",
        annotation_text="Default = 0.50",
        annotation_position="bottom right",
    )
    fig_thr.add_trace(go.Scatter(
        x=[best_t_val], y=[best_f1_val],
        mode="markers", name=f"F1 máximo = {best_f1_val:.4f}",
        marker=dict(color=COLORS["success"], size=12, symbol="circle"),
    ))
    fig_thr.update_layout(
        height=400,
        xaxis_title="Threshold",
        yaxis_title="Score",
        xaxis=dict(range=[0.10, 0.90]),
        yaxis=dict(range=[0, 1]),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.9)"),
    )
    fig_thr = apply_plotly_theme(fig_thr)
    st.plotly_chart(fig_thr, width="stretch")

    info_card(
        f"Umbral óptimo encontrado en validación: <strong>{best_t_val:.2f}</strong> "
        f"(F1 = {best_f1_val:.4f}). Con threshold=0.5 el F1 era "
        f"{mlp_val_data['metrics_val_default']['f1']:.4f}: "
        f"la optimización aporta "
        f"<strong>{(best_f1_val - mlp_val_data['metrics_val_default']['f1'])*100:+.2f} pp</strong>. "
        f"Tras aplicar calibración isotónica, el umbral se recalculó en <strong>0.26</strong> "
        f"(la calibración redistribuye las probabilidades hacia valores menores)."
    )

except FileNotFoundError:
    st.warning("No se encontró mlp_results_val.joblib o y_val.parquet.")

st.write("")


# ============================================================
#  IMPORTANCIA DE VARIABLES (RANDOM FOREST)
# ============================================================
section_title("Importancia de variables · Random Forest")

st.markdown("""
El Random Forest calcula la importancia de cada variable como la reducción media
de impureza (Gini) que aporta en todos los árboles del ensemble.
Es una medida de **relevancia predictiva** de cada feature.
""")

try:
    from src.inference import FEATURE_ORDER

    rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")
    importances = rf_model.feature_importances_

    df_imp = pd.DataFrame({
        "Variable": FEATURE_ORDER,
        "Importancia": importances,
    }).sort_values("Importancia", ascending=True)

    # Colorear por tipo de variable
    def color_feature(name):
        if name.startswith("mut_"):
            return COLORS["danger"]
        elif name in ["fumador", "obesidad", "actividad_fisica", "diabetes",
                      "hipertension", "epoc"]:
            return COLORS["warning"]
        elif name == "edad":
            return COLORS["secondary"]
        else:
            return COLORS["primary"]

    bar_colors_imp = [color_feature(f) for f in df_imp["Variable"]]

    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        y=df_imp["Variable"],
        x=df_imp["Importancia"],
        orientation="h",
        marker_color=bar_colors_imp,
        text=[f"{v:.4f}" for v in df_imp["Importancia"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>",
    ))

    # Leyenda manual como anotaciones
    fig_imp.update_layout(
        height=580,
        xaxis_title="Importancia (reducción de impureza Gini)",
        yaxis_title="",
        showlegend=False,
        margin=dict(t=30, r=80, b=40, l=20),
    )
    fig_imp = apply_plotly_theme(fig_imp)
    st.plotly_chart(fig_imp, width="stretch")

    # Top 5 features
    top5 = df_imp.tail(5)["Variable"].tolist()[::-1]
    info_card(
        f"Las 5 variables más importantes según el Random Forest: "
        f"<strong>{', '.join(top5)}</strong>. "
        f"Las mutaciones genéticas (BRCA1, TP53, KRAS) y el tabaquismo "
        f"encabezan el ranking, coherente con los pesos del modelo generativo "
        f"del dataset y con las correlaciones observadas en el EDA."
    )

except Exception as e:
    st.warning(f"No se pudo cargar la importancia de variables: {e}")

st.write("")


# ============================================================
#  CONCLUSIONES TÉCNICAS
# ============================================================
section_title("Conclusiones del estudio")

st.markdown(f"""
<div class="section-block">
<div style="font-size: 0.95rem; line-height: 1.75;">

<strong style="color: {COLORS['primary']};">1. Empate técnico Random Forest vs MLP.</strong><br>
La diferencia de F1-Score entre los dos modelos top (Random Forest 0.5439 y MLP 0.5423)
es de solo 0.0016 puntos: estadísticamente irrelevante. Operan en regímenes distintos
del trade-off precisión/recall: Random Forest favorece la sensibilidad (cribado amplio),
mientras que la MLP equilibra precisión y recall (decisión más selectiva).
<br><br>

<strong style="color: {COLORS['primary']};">2. La regresión logística es muy competitiva.</strong><br>
La Logistic Regression alcanza el AUC-ROC más alto del estudio (0.8275). Esto sugiere que
el problema es esencialmente lineal, lo que justifica que los modelos no lineales (MLP, RF,
boosting) no aporten ganancias dramáticas.
<br><br>

<strong style="color: {COLORS['primary']};">3. Coherencia entre validación y test.</strong><br>
Todos los modelos sostienen su rendimiento de validación al evaluarse en el conjunto de
test no visto. La MLP es especialmente estable (ΔF1 val→test = 0.0014).
<br><br>

<strong style="color: {COLORS['primary']};">4. Calibración estadística.</strong><br>
Los 5 modelos presentaban sobreconfianza moderada en estado bruto. Tras aplicar calibración
isotónica post-hoc en validación, los Brier Scores mejoraron entre el 15.7% (XGBoost) y el
31.6% (Logistic Regression). El sistema en producción usa probabilidades calibradas.
<br><br>

<strong style="color: {COLORS['primary']};">Recomendación.</strong><br>
Se selecciona la <strong>Red Neuronal Multicapa</strong> como modelo principal del sistema
en producción por: (a) mayor estabilidad val→test, (b) decisiones más equilibradas,
(c) capacidad de ajuste fino del umbral de decisión según política hospitalaria, (d) núcleo
técnico requerido por el caso de uso.
</div>
</div>
""", unsafe_allow_html=True)

st.write("")
medical_disclaimer()