"""
streamlit_app/pages/3_Analisis_Datos.py
=======================================
Página de análisis exploratorio del dataset clínico.

Muestra:
  - KPIs poblacionales y prevalencia oncológica
  - Variables incluidas y justificación de las excluidas
  - Correlaciones con el diagnóstico
  - Distribuciones interactivas por variable
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "streamlit_app"))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from assets.ui import (
    load_css, app_header, section_title, kpi_card,
    info_card, medical_disclaimer, COLORS, apply_plotly_theme,
)


# ============================================================
#  CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Análisis del dataset · Sistema Oncológico",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()


# ============================================================
#  CARGA DE DATOS (cacheado)
# ============================================================
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = PROJECT_ROOT / "data" / "raw"


@st.cache_data
def load_eda_summary():
    with open(DOCS_DIR / "eda_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_master_dataset():
    """Carga el dataset maestro mergeado para análisis."""
    from src.data_loader import load_master_dataset as _load
    return _load()


eda = load_eda_summary()
df = load_master_dataset()


# ============================================================
#  HEADER
# ============================================================
app_header(
    "Análisis del dataset clínico",
    "Exploración de la población base, variables seleccionadas y correlaciones con el diagnóstico oncológico",
)


# ============================================================
#  KPIs POBLACIONALES
# ============================================================
section_title("Población y desbalance de clases")

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Pacientes totales",
             f"{eda['n_total']:,}".replace(",", "."),
             "Población base del estudio")
with c2:
    kpi_card("Prevalencia oncológica",
             f"{eda['prevalencia']*100:.2f}%",
             "Pacientes con diagnóstico positivo")
with c3:
    n_cancer = int(eda['n_total'] * eda['prevalencia'])
    n_no_cancer = eda['n_total'] - n_cancer
    kpi_card("Ratio de desbalance",
             f"{n_no_cancer/n_cancer:.2f} : 1",
             "Sano vs cáncer")
with c4:
    kpi_card("Variables analizadas",
             f"{eda['n_features_finales']}",
             f"De {eda['n_features_finales']+17} originales")

st.write("")


# ============================================================
#  DISTRIBUCIÓN DE LA VARIABLE OBJETIVO
# ============================================================
section_title("Distribución del diagnóstico")

col_a, col_b = st.columns([1, 1.3])

with col_a:
    # Donut chart de la prevalencia
    fig_donut = go.Figure(data=[go.Pie(
        labels=["Sin cáncer", "Con cáncer"],
        values=[n_no_cancer, n_cancer],
        hole=0.55,
        marker=dict(colors=[COLORS["primary"], COLORS["danger"]]),
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=13),
    )])
    fig_donut.update_layout(
        height=350,
        showlegend=False,
        margin=dict(t=20, r=20, b=20, l=20),
        annotations=[dict(text=f"<b>{eda['n_total']:,}</b><br>pacientes".replace(",", "."),
                          x=0.5, y=0.5, font_size=14, showarrow=False)],
    )
    fig_donut = apply_plotly_theme(fig_donut)
    st.plotly_chart(fig_donut, width="stretch")

with col_b:
    st.write("")
    st.write("")
    st.markdown(f"""
La población del estudio presenta una prevalencia oncológica del
**{eda['prevalencia']*100:.2f}%**, lo que implica un
**desbalance de clases moderado** (ratio aproximado de
{n_no_cancer/n_cancer:.2f} pacientes sanos por cada caso de cáncer).
""")
    st.write("")
    st.markdown("""
Este desbalance se ha gestionado durante el entrenamiento mediante
**pesos de clase balanceados** (`{0: 0.62, 1: 2.59}`),
que penalizan más los errores en la clase minoritaria. Adicionalmente,
la **estratificación** en los tres conjuntos
(train/validación/test) garantiza que la prevalencia se mantiene
constante en cada uno (~19.3%).
""")
    st.write("")
    st.markdown("""
La prevalencia es **realista** respecto a la incidencia
poblacional de cáncer en grupos de riesgo, lo que da plausibilidad al
modelo generativo del dataset.
""")

st.write("")


# ============================================================
#  VARIABLES INCLUIDAS
# ============================================================
section_title("Variables incluidas en el modelo")

variables_incluidas = {
    "Bioquímica sanguínea (7)": [
        ("glucosa",       "Glucosa sérica (mg/dL)",          "Marcador de metabolismo"),
        ("colesterol",    "Colesterol total (mg/dL)",        "Marcador cardiovascular"),
        ("trigliceridos", "Triglicéridos (mg/dL)",           "Síndrome metabólico"),
        ("hemoglobina",   "Hemoglobina (g/dL)",              "Anemia / oxigenación"),
        ("leucocitos",    "Leucocitos (×10³/µL)",            "Inflamación / defensa"),
        ("plaquetas",     "Plaquetas (×10³/µL)",             "Coagulación"),
        ("creatinina",    "Creatinina (mg/dL)",              "Función renal"),
    ],
    "Marcadores genéticos (7)": [
        ("mut_BRCA1",  "Mutación BRCA1",  "Cáncer de mama y ovario"),
        ("mut_TP53",   "Mutación TP53",   "Múltiples tipos tumorales"),
        ("mut_EGFR",   "Mutación EGFR",   "Cáncer de pulmón"),
        ("mut_KRAS",   "Mutación KRAS",   "Cáncer de páncreas y colon"),
        ("mut_PIK3CA", "Mutación PIK3CA", "Cáncer de mama y colon"),
        ("mut_ALK",    "Mutación ALK",    "Cáncer de pulmón no microcítico"),
        ("mut_BRAF",   "Mutación BRAF",   "Melanoma y cáncer colorrectal"),
    ],
    "Comorbilidades (4)": [
        ("diabetes",     "Diabetes mellitus",     "Patología metabólica"),
        ("hipertension", "Hipertensión arterial", "Riesgo cardiovascular"),
        ("obesidad",     "Obesidad (IMC ≥ 30)",   "Factor de riesgo múltiple"),
        ("epoc",         "EPOC",                  "Enfermedad pulmonar obstructiva"),
    ],
    "Estilo de vida y demografía (3)": [
        ("fumador",          "Tabaquismo activo",      "Hábito carcinógeno"),
        ("actividad_fisica", "Actividad física (0-2)", "Factor protector"),
        ("edad",             "Edad (años)",            "Factor demográfico"),
    ],
}

tabs = st.tabs(list(variables_incluidas.keys()))
for tab, (categoria, vars_list) in zip(tabs, variables_incluidas.items()):
    with tab:
        df_cat = pd.DataFrame(vars_list, columns=["Variable", "Descripción", "Indicador clínico"])
        st.dataframe(df_cat, width="stretch", hide_index=True)

st.write("")


# ============================================================
#  VARIABLES EXCLUIDAS
# ============================================================
section_title("Variables excluidas y justificación")

variables_excluidas = [
    # Leakage (variables que solo se conocen a posteriori)
    ("coste_total",       "Leakage", "Coste se conoce solo después de tratamiento"),
    ("coste_farmaco",     "Leakage", "Lo mismo que coste_total"),
    ("num_ingresos",      "Leakage", "Los ingresos hospitalarios son consecuencia, no causa"),
    ("dias_hospital",     "Leakage", "Estancia hospitalaria post-diagnóstico"),
    ("tipo_seguro",       "Leakage", "Suele cambiar tras el diagnóstico"),
    ("vive",              "Leakage", "Variable de supervivencia, post-diagnóstico"),
    # Sin señal (correlaciones cercanas a 0 con el target)
    ("alcohol",                "Sin señal", "Variable constante en el dataset"),
    ("enfermedad_cardiaca",    "Sin señal", "Lift respecto a prevalencia base = 1.07x (no informativo)"),
    ("asma",                   "Sin señal", "Lift = 1.03x (no informativo)"),
    # Demográficas sin valor predictivo
    ("nivel_educativo",       "Demográfica sin señal", "Sin correlación con cáncer en el dataset"),
    ("nivel_ingresos",        "Demográfica sin señal", "Sin señal predictiva"),
    ("zona",                  "Demográfica sin señal", "Variable categórica sin patrón"),
    ("estado_civil",          "Demográfica sin señal", "Sin relación con el target"),
    ("num_hijos",             "Demográfica sin señal", "Sin correlación detectable"),
    ("distancia_hospital_km", "Demográfica sin señal", "Variable logística, no clínica"),
]

df_excl = pd.DataFrame(variables_excluidas, columns=["Variable", "Motivo", "Justificación"])

# Coloreamos por motivo
def color_motivo(val):
    if "Leakage" in val:
        return "background-color: #FEF2F2; color: #991B1B"
    elif "Sin señal" in val:
        return "background-color: #FEF3C7; color: #92400E"
    else:
        return "background-color: #EFF6FF; color: #1E3A8A"

st.dataframe(
    df_excl.style.map(color_motivo, subset=["Motivo"]),
    width="stretch",
    hide_index=True,
)

info_card(
    "Las exclusiones no son arbitrarias: cada variable descartada está justificada por "
    "<strong>data leakage</strong> (información solo disponible después del diagnóstico, "
    "que invalidaría la utilidad predictiva), <strong>ausencia de señal</strong> "
    "(correlación con el diagnóstico estadísticamente nula), o "
    "<strong>irrelevancia clínica</strong> (variables socioeconómicas o logísticas "
    "sin contribución al modelo). Esta selección reduce el ruido y mejora la "
    "interpretabilidad sin sacrificar capacidad predictiva."
)

st.write("")


# ============================================================
#  CORRELACIONES CON EL DIAGNÓSTICO
# ============================================================
section_title("Correlaciones con el diagnóstico")

# Calcular correlaciones de Pearson de cada feature numérica con cancer
features_for_corr = []
for cat, vars_list in variables_incluidas.items():
    for v, _, _ in vars_list:
        features_for_corr.append(v)

# La actividad física es categórica string, hay que mapear
df_corr = df.copy()
mapeo_actividad = {"Baja": 0, "Moderada": 1, "Alta": 2}
df_corr["actividad_fisica"] = df_corr["actividad_fisica"].map(mapeo_actividad).fillna(df_corr["actividad_fisica"])

correlations = []
for feat in features_for_corr:
    try:
        # Convertir a numérico forzosamente
        valores = pd.to_numeric(df_corr[feat], errors='coerce')
        corr = valores.corr(df_corr["cancer"])
        if not pd.isna(corr):
            correlations.append({"Variable": feat, "Correlación": corr})
    except Exception:
        continue

df_corrs = pd.DataFrame(correlations).sort_values("Correlación", ascending=True)

# Asignar color según signo y magnitud
bar_colors = []
for v in df_corrs["Correlación"]:
    if v >= 0.10:
        bar_colors.append(COLORS["danger"])  # factor de riesgo claro
    elif v >= 0.05:
        bar_colors.append(COLORS["warning"])
    elif v <= -0.05:
        bar_colors.append(COLORS["success"])  # factor protector
    else:
        bar_colors.append("#94A3B8")  # poca señal

fig_corr = go.Figure()
fig_corr.add_trace(go.Bar(
    y=df_corrs["Variable"],
    x=df_corrs["Correlación"],
    orientation="h",
    marker_color=bar_colors,
    text=[f"{v:+.3f}" for v in df_corrs["Correlación"]],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Correlación: %{x:+.4f}<extra></extra>",
))
fig_corr.add_vline(x=0, line_color="#1A202C", line_width=1)

fig_corr.update_layout(
    height=600,
    xaxis_title="Coeficiente de correlación de Pearson con el diagnóstico (cancer)",
    yaxis_title="",
    xaxis=dict(range=[-0.20, 0.30]),
    showlegend=False,
    margin=dict(t=30, r=60, b=40, l=40),
)
fig_corr = apply_plotly_theme(fig_corr)
st.plotly_chart(fig_corr, width="stretch")

info_card(
    "Las correlaciones más fuertes corresponden a las <strong>mutaciones genéticas</strong> "
    "(BRCA1, TP53, KRAS), seguidas del <strong>tabaquismo</strong> y la "
    "<strong>obesidad</strong>. La <strong>actividad física</strong> y la "
    "<strong>hemoglobina</strong> aparecen como factores protectores con correlación "
    "negativa. La mutación <strong>ALK</strong> (correlación ~0) confirma que el modelo "
    "no la asocia con cáncer en este dataset, validando que el modelo no alucina señales "
    "donde no las hay."
)

st.write("")


# ============================================================
#  DISTRIBUCIONES INTERACTIVAS
# ============================================================
section_title("Distribución por variable")

# Solo las features numéricas continuas (no las binarias)
features_continuas = ["glucosa", "colesterol", "trigliceridos", "hemoglobina",
                      "leucocitos", "plaquetas", "creatinina", "edad"]

selected_var = st.selectbox(
    "Seleccione una variable para ver su distribución según diagnóstico:",
    features_continuas,
    index=0,
)

col_dist, col_stats = st.columns([1.5, 1])

with col_dist:
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df[df["cancer"] == 0][selected_var],
        name="Sin cáncer",
        marker_color=COLORS["primary"],
        opacity=0.65,
        nbinsx=40,
    ))
    fig_dist.add_trace(go.Histogram(
        x=df[df["cancer"] == 1][selected_var],
        name="Con cáncer",
        marker_color=COLORS["danger"],
        opacity=0.65,
        nbinsx=40,
    ))
    fig_dist.update_layout(
        barmode="overlay",
        height=400,
        xaxis_title=selected_var,
        yaxis_title="Nº pacientes",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                    bgcolor="rgba(255,255,255,0.9)"),
    )
    fig_dist = apply_plotly_theme(fig_dist)
    st.plotly_chart(fig_dist, width="stretch")

with col_stats:
    media_no_cancer = df[df["cancer"] == 0][selected_var].mean()
    media_cancer = df[df["cancer"] == 1][selected_var].mean()
    diff_pct = (media_cancer - media_no_cancer) / media_no_cancer * 100

    st.markdown(
        f'<div style="font-size: 0.78rem; color: {COLORS["text_muted"]}; '
        f'text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px;">'
        f'Estadísticos descriptivos</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"**Sin cáncer**")
    st.caption(f"Media: {media_no_cancer:.2f} · Mediana: {df[df['cancer'] == 0][selected_var].median():.2f}")

    st.markdown(f"**Con cáncer**")
    st.caption(f"Media: {media_cancer:.2f} · Mediana: {df[df['cancer'] == 1][selected_var].median():.2f}")

    st.markdown(f"**Diferencia relativa**")
    color_diff = COLORS["danger"] if diff_pct > 0 else COLORS["success"]
    st.markdown(
        f'<div style="color: {color_diff}; font-weight: 600; font-size: 1rem;">'
        f'{diff_pct:+.2f}%</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Diferencia entre la media del grupo con cáncer y sin cáncer.")

st.write("")
medical_disclaimer()