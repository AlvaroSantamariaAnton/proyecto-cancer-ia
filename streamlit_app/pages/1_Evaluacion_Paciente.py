"""
streamlit_app/pages/1_Evaluacion_Paciente.py
============================================
Página de evaluación individual de paciente.

Formulario clínico → Predicción con los 5 modelos → Visualización del riesgo
con factores destacados, comparación poblacional y recomendaciones.
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
import plotly.graph_objects as go

from assets.ui import (
    load_css, app_header, section_title, kpi_card,
    risk_indicator, factor_chip, info_card, medical_disclaimer,
    COLORS, apply_plotly_theme,
)
from src.inference import (
    InferenceEngine, predict_patient, detect_risk_factors,
    RANGOS_NORMALES,
)


# ============================================================
#  CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Evaluación · Sistema Oncológico",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()


# ============================================================
#  CARGA DEL MOTOR DE INFERENCIA (cacheado entre sesiones)
# ============================================================
@st.cache_resource
def get_engine():
    """Carga los modelos una sola vez y los mantiene en memoria."""
    return InferenceEngine().load()


engine = get_engine()


# ============================================================
#  HEADER
# ============================================================
app_header(
    "Evaluación de paciente",
    "Introduzca los datos clínicos para obtener una predicción de riesgo oncológico",
)


# ============================================================
#  PERFILES DE EJEMPLO (botones rápidos para demo)
# ============================================================
PERFILES_EJEMPLO = {
    "Paciente sano": {
        # Sin mutaciones, no fumador, actividad alta, joven, bioquímica normal.
        # Esperado: riesgo BAJO (~3-5%), modelos coinciden en NO CÁNCER.
        "glucosa": 88, "colesterol": 180, "trigliceridos": 110, "hemoglobina": 14.5,
        "leucocitos": 6.5, "plaquetas": 250, "creatinina": 0.9,
        "mut_BRCA1": 0, "mut_TP53": 0, "mut_EGFR": 0, "mut_KRAS": 0,
        "mut_PIK3CA": 0, "mut_ALK": 0, "mut_BRAF": 0,
        "diabetes": 0, "hipertension": 0, "obesidad": 0, "epoc": 0,
        "fumador": 0, "actividad_fisica": "Alta", "edad": 35,
    },
    "Paciente intermedio": {
        # Solo factores de riesgo menores: hipertensión, ligero sobrepeso (sin
        # obesidad), ex-fumador (no fumador activo), edad media. Sin mutaciones.
        # Esperado: riesgo MODERADO (~30-50%), posible discordancia entre modelos.
        "glucosa": 105, "colesterol": 215, "trigliceridos": 175, "hemoglobina": 13.2,
        "leucocitos": 8.2, "plaquetas": 245, "creatinina": 1.0,
        "mut_BRCA1": 0, "mut_TP53": 0, "mut_EGFR": 0, "mut_KRAS": 0,
        "mut_PIK3CA": 0, "mut_ALK": 0, "mut_BRAF": 0,
        "diabetes": 0, "hipertension": 1, "obesidad": 0, "epoc": 0,
        "fumador": 0, "actividad_fisica": "Moderada", "edad": 58,
    },
    "Paciente con factores acumulados": {
        # Un solo gran factor (TP53) + fumador + hipertensión.
        # Esperado: riesgo ALTO (~70-85%), recomendación de derivación.
        "glucosa": 110, "colesterol": 210, "trigliceridos": 165, "hemoglobina": 12.8,
        "leucocitos": 8.5, "plaquetas": 240, "creatinina": 1.0,
        "mut_BRCA1": 0, "mut_TP53": 1, "mut_EGFR": 0, "mut_KRAS": 0,
        "mut_PIK3CA": 0, "mut_ALK": 0, "mut_BRAF": 0,
        "diabetes": 0, "hipertension": 1, "obesidad": 0, "epoc": 0,
        "fumador": 1, "actividad_fisica": "Moderada", "edad": 52,
    },
    "Paciente alto riesgo": {
        # Múltiples mutaciones graves + fumador + obesidad + bioquímica alterada.
        # Esperado: riesgo ALTO (>95%), consenso unánime de los 5 modelos.
        "glucosa": 145, "colesterol": 230, "trigliceridos": 220, "hemoglobina": 10.5,
        "leucocitos": 11.5, "plaquetas": 280, "creatinina": 1.0,
        "mut_BRCA1": 1, "mut_TP53": 1, "mut_EGFR": 0, "mut_KRAS": 1,
        "mut_PIK3CA": 0, "mut_ALK": 0, "mut_BRAF": 0,
        "diabetes": 1, "hipertension": 1, "obesidad": 1, "epoc": 0,
        "fumador": 1, "actividad_fisica": "Baja", "edad": 65,
    },
}


# Estado inicial del formulario (perfil sano por defecto)
if "patient_data" not in st.session_state:
    st.session_state["patient_data"] = PERFILES_EJEMPLO["Paciente sano"].copy()
if "evaluated" not in st.session_state:
    st.session_state["evaluated"] = False


def aplicar_perfil(nombre: str):
    st.session_state["patient_data"] = PERFILES_EJEMPLO[nombre].copy()
    st.session_state["evaluated"] = False


# ============================================================
#  SIDEBAR — Perfiles rápidos
# ============================================================
with st.sidebar:
    st.markdown("### Perfiles de ejemplo")
    st.caption("Cargue un caso típico para evaluación rápida.")
    for nombre in PERFILES_EJEMPLO:
        st.button(nombre, width="stretch",
                  on_click=aplicar_perfil, args=(nombre,))

    st.markdown("---")
    st.markdown("### Información del modelo")
    st.markdown(f"""
    <div style="font-size: 0.83rem; color: {COLORS['text_muted']}; line-height: 1.6;">
    <strong>Modelo principal:</strong> Red Neuronal Multicapa<br>
    <strong>Umbral de decisión:</strong> {engine.mlp_threshold:.2f}<br>
    <strong>Modelos secundarios:</strong> 4 algoritmos clásicos<br>
    <strong>Variables analizadas:</strong> 21 features<br>
    <strong>Calibración:</strong> Isotónica post-hoc
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  FORMULARIO CLÍNICO
# ============================================================
section_title("Datos del paciente")

current = st.session_state["patient_data"]

with st.form("patient_form"):
    # === Tabs por categoría clínica ===
    tab1, tab2, tab3, tab4 = st.tabs([
        "Bioquímica sanguínea",
        "Marcadores genéticos",
        "Comorbilidades",
        "Estilo de vida y demografía",
    ])

    # --- TAB 1: Bioquímica ---
    with tab1:
        st.caption("Valores de la analítica sanguínea más reciente")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            glucosa = st.number_input("Glucosa (mg/dL)",
                min_value=55.0, max_value=180.0,
                value=float(current["glucosa"]), step=1.0,
                help=f"Normal: {RANGOS_NORMALES['glucosa']['min']}-{RANGOS_NORMALES['glucosa']['max']} mg/dL · Diabetes: >{RANGOS_NORMALES['glucosa']['umbral_alto']}")
            colesterol = st.number_input("Colesterol total (mg/dL)",
                min_value=120.0, max_value=320.0,
                value=float(current["colesterol"]), step=1.0,
                help="Normal: <200 mg/dL · Alto: ≥240")
        with c2:
            trigliceridos = st.number_input("Triglicéridos (mg/dL)",
                min_value=50.0, max_value=325.0,
                value=float(current["trigliceridos"]), step=1.0,
                help="Normal: <150 mg/dL · Alto: ≥200")
            hemoglobina = st.number_input("Hemoglobina (g/dL)",
                min_value=8.0, max_value=18.0,
                value=float(current["hemoglobina"]), step=0.1,
                help="Normal: 12-16 g/dL · Anemia: <11")
        with c3:
            leucocitos = st.number_input("Leucocitos (×10³/µL)",
                min_value=2.0, max_value=15.0,
                value=float(current["leucocitos"]), step=0.1,
                help="Normal: 4.5-11 · Inflamación: >10")
            plaquetas = st.number_input("Plaquetas (×10³/µL)",
                min_value=100.0, max_value=500.0,
                value=float(current["plaquetas"]), step=1.0,
                help="Normal: 150-400")
        with c4:
            creatinina = st.number_input("Creatinina (mg/dL)",
                min_value=0.35, max_value=2.10,
                value=float(current["creatinina"]), step=0.01,
                help="Normal: 0.6-1.3 mg/dL · Marcador de función renal")

    # --- TAB 2: Marcadores genéticos ---
    with tab2:
        st.caption("Mutaciones oncogénicas detectadas en el estudio genético")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            mut_BRCA1 = st.checkbox("BRCA1 (mama, ovario)",
                value=bool(current["mut_BRCA1"]))
            mut_TP53 = st.checkbox("TP53 (múltiples tipos)",
                value=bool(current["mut_TP53"]))
        with c2:
            mut_KRAS = st.checkbox("KRAS (páncreas, colon)",
                value=bool(current["mut_KRAS"]))
            mut_EGFR = st.checkbox("EGFR (pulmón)",
                value=bool(current["mut_EGFR"]))
        with c3:
            mut_PIK3CA = st.checkbox("PIK3CA (mama, colon)",
                value=bool(current["mut_PIK3CA"]))
            mut_BRAF = st.checkbox("BRAF (melanoma)",
                value=bool(current["mut_BRAF"]))
        with c4:
            mut_ALK = st.checkbox("ALK (pulmón no microcítico)",
                value=bool(current["mut_ALK"]))

    # --- TAB 3: Comorbilidades ---
    with tab3:
        st.caption("Patologías crónicas diagnosticadas")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            diabetes = st.checkbox("Diabetes mellitus",
                value=bool(current["diabetes"]))
        with c2:
            hipertension = st.checkbox("Hipertensión arterial",
                value=bool(current["hipertension"]))
        with c3:
            obesidad = st.checkbox("Obesidad (IMC ≥ 30)",
                value=bool(current["obesidad"]))
        with c4:
            epoc = st.checkbox("EPOC",
                value=bool(current["epoc"]))

    # --- TAB 4: Estilo de vida y demografía ---
    with tab4:
        st.caption("Hábitos de vida y datos demográficos")
        c1, c2, c3 = st.columns(3)
        with c1:
            fumador = st.radio("Tabaquismo activo",
                ["No", "Sí"],
                index=int(current["fumador"]),
                horizontal=True)
        with c2:
            af_options = ["Baja", "Moderada", "Alta"]
            actividad_fisica = st.radio("Actividad física habitual",
                af_options,
                index=af_options.index(current["actividad_fisica"]),
                horizontal=True,
                help="Baja=sedentaria · Moderada=2-3 sesiones/semana · Alta=≥4 sesiones/semana")
        with c3:
            edad = st.number_input("Edad (años)",
                min_value=20, max_value=90,
                value=int(current["edad"]), step=1)

    st.write("")
    submitted = st.form_submit_button(
        "Evaluar paciente",
        type="primary",
        use_container_width=False,
    )


# ============================================================
#  PROCESAR EVALUACIÓN
# ============================================================
if submitted:
    patient = {
        "glucosa": glucosa, "colesterol": colesterol, "trigliceridos": trigliceridos,
        "hemoglobina": hemoglobina, "leucocitos": leucocitos, "plaquetas": plaquetas,
        "creatinina": creatinina,
        "mut_BRCA1": int(mut_BRCA1), "mut_TP53": int(mut_TP53),
        "mut_EGFR": int(mut_EGFR), "mut_KRAS": int(mut_KRAS),
        "mut_PIK3CA": int(mut_PIK3CA), "mut_ALK": int(mut_ALK),
        "mut_BRAF": int(mut_BRAF),
        "diabetes": int(diabetes), "hipertension": int(hipertension),
        "obesidad": int(obesidad), "epoc": int(epoc),
        "fumador": 1 if fumador == "Sí" else 0,
        "actividad_fisica": actividad_fisica, "edad": edad,
    }
    st.session_state["patient_data"] = patient
    st.session_state["evaluated"] = True
    st.session_state["last_result"] = predict_patient(engine, patient)


# ============================================================
#  RESULTADOS (solo si se ha evaluado)
# ============================================================
if st.session_state["evaluated"]:
    result = st.session_state["last_result"]
    patient = st.session_state["patient_data"]

    st.write("")
    st.markdown("---")
    section_title("Resultado de la evaluación")

    # === Indicador de riesgo principal y resumen lateral ===
    col_risk, col_recap = st.columns([1.4, 1])

    with col_risk:
        # Usamos la probabilidad de display (capada [0.02, 0.98]) para el indicador
        risk_indicator(result["risk_level"], result["mlp_probability_display"])

        # Recomendación clínica según nivel
        if result["risk_level"] == "Alto":
            st.markdown(f"""
            <div class="info-card" style="background: #FEF2F2; border-color: {COLORS['danger']}; color: #991B1B;">
            <strong>Recomendación:</strong> Derivación urgente a consulta especializada para pruebas
            diagnósticas confirmatorias (analítica completa, biopsia o estudios de imagen según
            sospecha clínica). Considerar revisión de antecedentes familiares oncológicos.
            </div>
            """, unsafe_allow_html=True)
        elif result["risk_level"] == "Moderado":
            st.markdown(f"""
            <div class="info-card" style="background: #FFFBEB; border-color: {COLORS['warning']}; color: #92400E;">
            <strong>Recomendación:</strong> Seguimiento clínico estrecho. Programar pruebas adicionales
            según sospecha diagnóstica y revisar factores de riesgo modificables (tabaquismo, peso,
            actividad física, control glucémico).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-card" style="background: #F0FDF4; border-color: {COLORS['success']}; color: #065F46;">
            <strong>Recomendación:</strong> Riesgo bajo según el modelo. Mantener controles preventivos
            habituales según protocolo poblacional. Reforzar hábitos saludables.
            </div>
            """, unsafe_allow_html=True)

    with col_recap:
        st.markdown(f"""
        <div class="section-block" style="margin-top: 0;">
            <div style="font-size: 0.78rem; color: {COLORS['text_muted']};
                        text-transform: uppercase; letter-spacing: 0.06em;">
                Probabilidad estimada
            </div>
            <div style="font-size: 2.6rem; font-weight: 700; color: {COLORS['primary']};
                        margin: 6px 0; line-height: 1;">
                {result['mlp_probability_display']*100:.1f}%
            </div>
            <div style="font-size: 0.85rem; color: {COLORS['text_muted']}; line-height: 1.55;">
                Modelo principal: Red Neuronal Multicapa<br>
                Umbral de decisión: {result['mlp_threshold']:.2f}<br>
                Decisión: <strong style="color: {COLORS['text']};">{'Sospecha de cáncer' if result['mlp_prediction'] == 1 else 'Sin sospecha'}</strong><br>
                <em style="font-size: 0.78rem;">Estimación calibrada estadísticamente.<br>
                No constituye un diagnóstico.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # === Comparación entre los 5 modelos ===
    section_title("Concordancia entre modelos")

    # Capamos todas las probabilidades a [0.02, 0.98] para coherencia visual
    df_probs = pd.DataFrame([
        {"Modelo": name,
         "Probabilidad": max(0.02, min(0.98, prob)),
         "Predicción": "Cáncer" if result["predictions"][name] == 1 else "No cáncer",
         "Threshold": result["thresholds"][name]}
        for name, prob in result["probabilities"].items()
    ])

    # La MLP la sobrescribimos con su probabilidad de display (capada [0.02, 0.98])
    mlp_name = next(n for n in result["probabilities"] if "MLP" in n)
    df_probs.loc[df_probs["Modelo"] == mlp_name, "Probabilidad"] = result["mlp_probability_display"]

    df_probs = df_probs.sort_values("Probabilidad", ascending=True)

    fig = go.Figure()
    bar_colors = [COLORS["primary"] if "MLP" in m else "#94A3B8"
                  for m in df_probs["Modelo"]]

    fig.add_trace(go.Bar(
        y=df_probs["Modelo"], x=df_probs["Probabilidad"],
        orientation="h", marker_color=bar_colors,
        text=[f"{p*100:.1f}%" for p in df_probs["Probabilidad"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Probabilidad: %{x:.4f}<extra></extra>",
    ))
    # Líneas verticales con los thresholds
    fig.add_vline(x=0.5, line_dash="dot", line_color="#94A3B8",
                  annotation_text="t=0.5 (modelos clásicos)",
                  annotation_position="top")
    fig.add_vline(x=engine.mlp_threshold, line_dash="dash", line_color=COLORS["primary"],
                  annotation_text=f"t={engine.mlp_threshold:.2f} (MLP)",
                  annotation_position="bottom")

    fig.update_layout(
        height=300,
        xaxis=dict(range=[0, 1.05], tickformat=".0%"),
        xaxis_title="Probabilidad estimada",
        showlegend=False,
        margin=dict(t=40, r=20, b=40, l=20),
    )
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, width="stretch")

    # Lectura del consenso
    n_positivos = sum(1 for v in result["predictions"].values() if v == 1)
    n_total = len(result["predictions"])
    if n_positivos == n_total:
        consenso_msg = f"Los 5 modelos coinciden en señalar al paciente como <strong>caso sospechoso</strong> (consenso unánime)."
    elif n_positivos == 0:
        consenso_msg = f"Los 5 modelos coinciden en clasificar al paciente como <strong>no sospechoso</strong>."
    else:
        consenso_msg = f"Discordancia entre modelos: {n_positivos} de {n_total} predicen cáncer. Considerar evaluación clínica adicional."
    info_card(consenso_msg)

    st.write("")

    # === Factores de riesgo detectados ===
    section_title("Factores de riesgo identificados")

    factores = detect_risk_factors(patient)

    if not factores:
        st.markdown(f"""
        <div style="padding: 16px; background: {COLORS['surface']};
                    border-radius: 6px; color: {COLORS['text_muted']};">
            No se han detectado factores de riesgo significativos en los datos introducidos.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Agrupamos por severidad
        chips_html = ""
        for f in factores:
            chips_html += factor_chip(f["descripcion"], f["severidad"])
        st.markdown(chips_html, unsafe_allow_html=True)

        # Tabla detallada con pesos
        with st.expander("Ver detalle de pesos en el modelo de referencia"):
            df_factores = pd.DataFrame(factores)
            df_factores = df_factores[["descripcion", "severidad", "peso"]]
            df_factores.columns = ["Factor", "Severidad", "Peso (modelo generativo)"]
            st.dataframe(df_factores, width="stretch", hide_index=True)
            st.caption("Los pesos provienen del modelo logístico generativo del dataset (metadata). "
                       "Valores negativos indican factores protectores.")

    st.write("")

    # === Comparación con poblaciones de referencia (radar chart) ===
    section_title("Comparación con poblaciones de referencia")
    st.caption("Cómo se posiciona el paciente respecto a la media de pacientes con y sin diagnóstico oncológico.")

    # Seleccionamos las features bioquímicas + edad para el radar
    radar_features = ["glucosa", "colesterol", "trigliceridos",
                      "hemoglobina", "leucocitos", "edad"]
    paciente_vals = [patient[f] for f in radar_features]
    media_no_cancer = [engine.population_stats[f]["media_no_cancer"] for f in radar_features]
    media_cancer = [engine.population_stats[f]["media_cancer"] for f in radar_features]

    # Normalizamos al rango [0, 1] para que el radar sea comparable
    def normalize(values, feat_list):
        normalized = []
        for v, f in zip(values, feat_list):
            mean = engine.population_stats[f]["media_global"]
            std = engine.population_stats[f]["std_global"]
            # z-score reescalado a [0, 1] aproximado (centrado en 0.5)
            z = (v - mean) / std
            normalized.append(0.5 + z * 0.15)  # multiplicador para amplitud visual
        return normalized

    # Cerramos el polígono repitiendo el primer valor al final
    theta_closed = radar_features + [radar_features[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=normalize(media_no_cancer, radar_features) + [normalize(media_no_cancer, radar_features)[0]],
        theta=theta_closed,
        fill="toself",
        name="Población sin cáncer",
        line=dict(color=COLORS["success"]),
        fillcolor="rgba(16, 185, 129, 0.15)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=normalize(media_cancer, radar_features) + [normalize(media_cancer, radar_features)[0]],
        theta=theta_closed,
        fill="toself",
        name="Población con cáncer",
        line=dict(color=COLORS["danger"]),
        fillcolor="rgba(220, 38, 38, 0.15)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=normalize(paciente_vals, radar_features) + [normalize(paciente_vals, radar_features)[0]],
        theta=theta_closed,
        fill="toself",
        name="Paciente",
        line=dict(color=COLORS["primary"], width=3),
        fillcolor="rgba(0, 102, 204, 0.20)",
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(linecolor="#CBD5E1"),
            bgcolor="white",
        ),
        showlegend=True,
        height=420,
        margin=dict(t=30, r=20, b=20, l=20),
    )
    fig_radar = apply_plotly_theme(fig_radar)
    st.plotly_chart(fig_radar, width="stretch")

    st.write("")

    # === Datos del paciente evaluado (resumen final) ===
    with st.expander("Ver datos completos del paciente evaluado"):
        df_paciente = pd.DataFrame([{
            "Variable": k,
            "Valor introducido": str(v),  # Forzar string para evitar mezclas de tipo en Arrow
            "Media en población sin cáncer": f"{engine.population_stats[k]['media_no_cancer']:.2f}" if k in engine.population_stats else "—",
            "Media en población con cáncer": f"{engine.population_stats[k]['media_cancer']:.2f}" if k in engine.population_stats else "—",
        } for k, v in patient.items()])
        st.dataframe(df_paciente, width="stretch", hide_index=True)

    st.write("")
    medical_disclaimer()

else:
    # Mensaje si no se ha evaluado todavía
    st.write("")
    info_card(
        "Complete los datos del paciente en las pestañas superiores y pulse "
        "<strong>Evaluar paciente</strong> para obtener el resultado. Puede cargar perfiles "
        "de ejemplo desde el menú lateral."
    )