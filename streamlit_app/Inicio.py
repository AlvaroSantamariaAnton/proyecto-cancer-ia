"""
streamlit_app/Inicio.py
=======================
Página de inicio del Sistema de Evaluación de Riesgo Oncológico.

Panel ejecutivo con KPIs principales, navegación a las herramientas
y resumen institucional del estudio.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "streamlit_app"))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

from assets.ui import (
    load_css, app_header, section_title, kpi_card,
    info_card, medical_disclaimer, COLORS, apply_plotly_theme,
)


# ============================================================
#  CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Inicio · Sistema de Evaluación Oncológica",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()


# ============================================================
#  CARGA DE DATOS DEL ESTUDIO (cacheado)
# ============================================================
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


@st.cache_data
def load_results_test():
    """Carga la tabla comparativa final en test."""
    return pd.read_csv(REPORTS_DIR / "comparativa_final_test.csv")


@st.cache_data
def load_eda_summary():
    """Carga el resumen del EDA."""
    import json
    with open(PROJECT_ROOT / "docs" / "eda_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


df_results = load_results_test()
eda = load_eda_summary()


# ============================================================
#  HEADER
# ============================================================
app_header(
    "Sistema de Evaluación de Riesgo Oncológico",
    "Apoyo a la decisión clínica basado en aprendizaje automático · Versión 1.0",
)

# ============================================================
#  PANEL DE KPIs PRINCIPALES
# ============================================================
section_title("Rendimiento del sistema")

best_row = df_results.iloc[0]
mlp_row = df_results[df_results["Modelo"].str.startswith("MLP")].iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card(
        "Pacientes evaluados",
        f"{eda['n_total']:,}".replace(",", "."),
        f"Conjunto de validación clínica",
    )
with c2:
    kpi_card(
        "F1-Score (mejor modelo)",
        f"{best_row['F1-Score']:.4f}",
        f"{best_row['Modelo']}",
    )
with c3:
    kpi_card(
        "AUC-ROC del sistema",
        f"{mlp_row['AUC-ROC']:.4f}",
        "Red Neuronal Multicapa",
    )
with c4:
    kpi_card(
        "Sensibilidad operativa",
        f"{best_row['Recall']*100:.1f}%",
        f"Detección sobre conjunto de test",
    )

st.write("")

# ============================================================
#  RESUMEN DEL ESTUDIO
# ============================================================
col_left, col_right = st.columns([1.4, 1])

with col_left:
    section_title("Comparativa de modelos en test")

    # Gráfico horizontal de F1-Score por modelo
    fig = go.Figure()

    df_sorted = df_results.sort_values("F1-Score", ascending=True)
    is_mlp = df_sorted["Modelo"].str.startswith("MLP")
    bar_colors = [COLORS["primary"] if mlp else "#94A3B8"
                  for mlp in is_mlp]

    fig.add_trace(go.Bar(
        y=df_sorted["Modelo"],
        x=df_sorted["F1-Score"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.4f}" for v in df_sorted["F1-Score"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>F1-Score: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        height=320,
        title=dict(text="F1-Score sobre conjunto de test (n=10.001)",
                   font=dict(size=13, color=COLORS["text"])),
        xaxis_title="F1-Score",
        yaxis_title="",
        xaxis=dict(range=[0.50, 0.56]),
        showlegend=False,
    )
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    section_title("Características del estudio")

    st.markdown(f"""
    <div style="font-size: 0.92rem; line-height: 1.7;">
    <strong>Población base:</strong> {eda['n_total']:,} pacientes<br>
    <strong>Prevalencia oncológica:</strong> {eda['prevalencia']*100:.2f}%<br>
    <strong>Variables analizadas:</strong> {eda['n_features_finales']} clínicas<br>
    <strong>Modelos evaluados:</strong> 5 algoritmos<br>
    <strong>Conjunto de validación:</strong> 10.001 pacientes<br>
    <strong>Estratificación:</strong> Train/Validación/Test (60/20/20)
    </div>
    """.replace(",", "."), unsafe_allow_html=True)

st.write("")

# ============================================================
#  HERRAMIENTAS DISPONIBLES
# ============================================================
section_title("Herramientas disponibles")

tools = [
    {
        "name": "Evaluación de paciente",
        "desc": "Introduzca los datos clínicos, bioquímicos y genéticos del paciente para obtener una evaluación de riesgo oncológico. El sistema aplica los 5 modelos disponibles y proporciona una recomendación basada en el modelo de mayor rendimiento.",
        "page": "1_Evaluacion_Paciente",
    },
    {
        "name": "Comparativa técnica de modelos",
        "desc": "Análisis detallado del rendimiento de los modelos: métricas de validación, curvas ROC y Precisión-Recall, matrices de confusión y ranking final sobre el conjunto de test.",
        "page": "2_Comparativa_Modelos",
    },
    {
        "name": "Análisis del dataset",
        "desc": "Exploración estadística de la población base: prevalencias, distribuciones por variable, correlaciones con el diagnóstico, y justificación de las decisiones de inclusión/exclusión de variables.",
        "page": "3_Analisis_Datos",
    },
    {
        "name": "Metodología y limitaciones",
        "desc": "Documentación del pipeline analítico: arquitectura de la red neuronal, optimización del umbral de clasificación, gestión del desbalance de clases y advertencias sobre el uso clínico.",
        "page": "4_Metodologia",
    },
]

for tool in tools:
    st.markdown(f"""
    <div class="section-block">
        <div style="font-size: 1rem; font-weight: 600; color: {COLORS['primary']}; margin-bottom: 6px;">
            {tool['name']}
        </div>
        <div style="font-size: 0.9rem; color: {COLORS['text_muted']}; line-height: 1.55;">
            {tool['desc']}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
#  PIE DE PÁGINA INSTITUCIONAL
# ============================================================
st.write("")
medical_disclaimer()

st.markdown(f"""
<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid {COLORS['border']};
            font-size: 0.78rem; color: {COLORS['text_muted']}; text-align: center;">
Sistema de Evaluación de Riesgo Oncológico · Universidad Alfonso X el Sabio<br>
Asignatura de Inteligencia Artificial · Ingeniería Matemática · Curso 2025/2026<br>
Desarrollado por Álvaro Santamaría Antón
</div>
""", unsafe_allow_html=True)