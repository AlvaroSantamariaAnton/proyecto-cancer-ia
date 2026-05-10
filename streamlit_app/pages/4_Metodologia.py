"""
streamlit_app/pages/4_Metodologia.py
====================================
Documentación metodológica del sistema.

Pipeline analítico, arquitectura del modelo, decisiones técnicas
y limitaciones. Página orientada a revisión técnica.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "streamlit_app"))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import plotly.graph_objects as go

from assets.ui import (
    load_css, app_header, section_title, kpi_card,
    info_card, medical_disclaimer, COLORS, apply_plotly_theme,
)


# ============================================================
#  CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Metodología · Sistema Oncológico",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()


# ============================================================
#  HEADER
# ============================================================
app_header(
    "Metodología y limitaciones",
    "Documentación del pipeline analítico, arquitectura del modelo y advertencias sobre uso clínico",
)

info_card(
    "Esta página documenta las decisiones metodológicas y técnicas adoptadas durante "
    "el desarrollo del sistema. Su objetivo es facilitar la revisión técnica y la "
    "comprensión del pipeline, no actuar como manual de usuario."
)

st.write("")


# ============================================================
#  PIPELINE ANALÍTICO
# ============================================================
section_title("Pipeline analítico completo")

fases = [
    {
        "n": "01", "titulo": "Análisis Exploratorio (EDA)",
        "desc": "Carga de 6 ficheros CSV con datos clínicos, bioquímicos, genéticos y demográficos. "
                "Merge limpio sin duplicados ni nulos sobre 50.001 pacientes y 38 columnas iniciales. "
                "Validación de las distribuciones contra la metadata del dataset.",
        "outputs": "8 figuras del EDA, eda_summary.json, 21 features finales seleccionadas",
    },
    {
        "n": "02", "titulo": "Preprocesado y división de datos",
        "desc": "División estratificada 60/20/20 (train/val/test) preservando la prevalencia del 19.3% "
                "en cada subconjunto. Estandarización (StandardScaler) ajustada únicamente sobre el "
                "conjunto de entrenamiento para evitar data leakage. Codificación ordinal de la "
                "actividad física (Baja=0, Moderada=1, Alta=2). Cálculo de pesos de clase balanceados.",
        "outputs": "X_{train,val,test}.parquet · y_{train,val,test}.parquet · scaler.joblib",
    },
    {
        "n": "03", "titulo": "Modelos clásicos de Machine Learning",
        "desc": "Entrenamiento de 4 algoritmos clásicos sobre el train set: Logistic Regression, "
                "Random Forest, XGBoost y LightGBM. Configuración consistente con scale_pos_weight "
                "para gestionar el desbalance. Evaluación inicial sobre validación con threshold=0.5.",
        "outputs": "4 modelos persistidos · ranking de F1-Score en validación",
    },
    {
        "n": "04", "titulo": "Red Neuronal Multicapa (MLP)",
        "desc": "Diseño y entrenamiento del núcleo técnico del sistema. Arquitectura de 3 capas ocultas "
                "(256 → 128 → 64 neuronas) con BatchNormalization + ReLU + Dropout. Optimizador Adam, "
                "binary crossentropy, callbacks de EarlyStopping (patience=12) y ReduceLROnPlateau "
                "(factor=0.5, patience=6). Class weights {0: 0.62, 1: 2.59}.",
        "outputs": "mlp_cancer.keras (48.641 parámetros) · curvas de loss y accuracy",
    },
    {
        "n": "05", "titulo": "Optimización del umbral de decisión",
        "desc": "Búsqueda exhaustiva del threshold óptimo en validación maximizando F1-Score "
                "(rango [0.10, 0.90], paso 0.01). El umbral encontrado se aplica una sola vez al "
                "conjunto de test, manteniendo la integridad del protocolo de evaluación.",
        "outputs": "threshold_optimal=0.68 (cruda) · 0.26 (post-calibración)",
    },
    {
        "n": "06", "titulo": "Calibración isotónica post-hoc",
        "desc": "Auditoría de la calibración del modelo principal mediante diagrama de fiabilidad y "
                "Brier Score. Identificación de sobreconfianza moderada en la zona [0.6, 0.95]. "
                "Aplicación de regresión isotónica entrenada sobre validación a los 5 modelos para "
                "corregir las probabilidades. Reducción del Brier Score entre el 15% y el 32%.",
        "outputs": "mlp_calibrator.joblib · ml_calibrators.joblib",
    },
]

for fase in fases:
    st.markdown(f"""
    <div class="section-block" style="margin-bottom: 14px;">
        <div style="display: flex; align-items: flex-start; gap: 16px;">
            <div style="background: {COLORS['primary']}; color: white;
                        width: 44px; height: 44px; border-radius: 8px;
                        display: flex; align-items: center; justify-content: center;
                        font-weight: 700; font-size: 0.95rem; flex-shrink: 0;">
                {fase['n']}
            </div>
            <div style="flex: 1;">
                <div style="font-size: 1.05rem; font-weight: 600; color: {COLORS['text']};
                            margin-bottom: 6px;">
                    {fase['titulo']}
                </div>
                <div style="font-size: 0.9rem; color: {COLORS['text_muted']};
                            line-height: 1.6; margin-bottom: 8px;">
                    {fase['desc']}
                </div>
                <div style="font-size: 0.82rem; color: {COLORS['primary']};
                            font-family: monospace;">
                    Outputs: {fase['outputs']}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ============================================================
#  ARQUITECTURA DE LA RED NEURONAL
# ============================================================
section_title("Arquitectura de la red neuronal")

c1, c2 = st.columns([1.3, 1])

with c1:
    # Visualización esquemática de la arquitectura
    fig_arch = go.Figure()

    # Definimos las capas: posición x, número de neuronas representadas, etiqueta
    layers = [
        {"x": 0.05, "size": 21,  "label": "Input<br>(21 features)",         "color": COLORS["text_muted"]},
        {"x": 0.27, "size": 256, "label": "Dense<br>256 neuronas",         "color": COLORS["primary"]},
        {"x": 0.50, "size": 128, "label": "Dense<br>128 neuronas",         "color": COLORS["primary"]},
        {"x": 0.73, "size": 64,  "label": "Dense<br>64 neuronas",          "color": COLORS["primary"]},
        {"x": 0.95, "size": 1,   "label": "Output<br>sigmoid",             "color": COLORS["danger"]},
    ]

    # Conexiones entre capas (líneas)
    for i in range(len(layers) - 1):
        fig_arch.add_shape(
            type="line",
            x0=layers[i]["x"] + 0.04, x1=layers[i+1]["x"] - 0.04,
            y0=0.5, y1=0.5,
            line=dict(color="#CBD5E1", width=2),
        )

    # Capas como círculos con tamaño proporcional
    max_size = max(l["size"] for l in layers)
    for layer in layers:
        radius = 0.04 + 0.04 * (layer["size"] / max_size)
        fig_arch.add_shape(
            type="circle",
            x0=layer["x"] - radius, x1=layer["x"] + radius,
            y0=0.5 - radius * 2, y1=0.5 + radius * 2,
            fillcolor=layer["color"],
            line=dict(color=layer["color"], width=2),
            opacity=0.8,
        )
        fig_arch.add_annotation(
            x=layer["x"], y=0.15,
            text=layer["label"],
            showarrow=False,
            font=dict(size=11, color=COLORS["text"]),
        )

    fig_arch.update_layout(
        height=280,
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(range=[0, 1], visible=False),
        showlegend=False,
        margin=dict(t=20, r=20, b=20, l=20),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_arch, width="stretch")

    st.caption("Esquema de la arquitectura. Entre cada capa Dense hay BatchNormalization, "
               "activación ReLU y Dropout (0.25 / 0.25 / 0.20).")

with c2:
    st.markdown(f"""
    <div class="section-block" style="margin-top: 0;">
        <div style="font-size: 0.78rem; color: {COLORS['text_muted']};
                    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px;">
            Especificaciones técnicas
        </div>
        <div style="font-size: 0.9rem; line-height: 1.8;">
            <strong>Parámetros totales:</strong> 48.641<br>
            <strong>Parámetros entrenables:</strong> 47.745<br>
            <strong>Optimizador:</strong> Adam (lr=1e-3)<br>
            <strong>Función de pérdida:</strong> Binary crossentropy<br>
            <strong>Batch size:</strong> 64<br>
            <strong>Épocas máximas:</strong> 100<br>
            <strong>Épocas reales:</strong> 42 (EarlyStopping)<br>
            <strong>Mejor época:</strong> 30<br>
            <strong>Tiempo de entrenamiento:</strong> ~2 min (CPU)
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ============================================================
#  GESTIÓN DEL DESBALANCE
# ============================================================
section_title("Gestión del desbalance de clases")

st.markdown("""
La prevalencia oncológica en el dataset es del **19.3%**, lo que produce un desbalance
moderado de clases (4.18 pacientes sanos por cada caso de cáncer). Sin tratamiento específico,
los modelos tenderían a predecir mayoritariamente la clase negativa, ignorando la
clase minoritaria que es justamente la de mayor interés clínico.

**Estrategia adoptada:** ponderación de clases (`class_weight`) inversamente proporcional
a su frecuencia. El cálculo de scikit-learn devuelve `{0: 0.62, 1: 2.59}`, lo que implica
que cada error sobre un caso real de cáncer pesa 4.2 veces más que un error sobre un sano
durante el cálculo de la pérdida.

**Por qué no SMOTE:** se descartó el oversampling sintético para evitar
introducir muestras artificiales en un problema clínico. La ponderación de clases logra
el mismo objetivo (atender a la clase minoritaria) sin alterar la distribución original.
""")

st.write("")


# ============================================================
#  OPTIMIZACIÓN DEL UMBRAL
# ============================================================
section_title("Optimización del umbral de decisión")

st.markdown("""
La probabilidad de salida de un modelo (entre 0 y 1) se convierte en una decisión binaria
(cáncer / no cáncer) mediante un **umbral de decisión**. El valor por defecto de 0.5
no siempre es óptimo en problemas con clases desbalanceadas.

**Procedimiento:**

1. Para cada threshold candidato del rango `[0.10, 0.90]` con paso de `0.01`, se calcula
   el F1-Score sobre el conjunto de **validación** (no test).
2. Se selecciona el threshold que maximiza el F1.
3. Ese umbral se aplica **una sola vez** al conjunto de test para el reporte final,
   garantizando la integridad del protocolo experimental.

**Resultado para la MLP:** el threshold óptimo en validación con probabilidades crudas
fue `0.68`, y tras aplicar la calibración isotónica el threshold se recalculó en `0.26`
(coherente, ya que la calibración redistribuye las probabilidades hacia valores menores).
""")

st.write("")


# ============================================================
#  CALIBRACIÓN
# ============================================================
section_title("Calibración isotónica post-hoc")

st.markdown("""
Las redes neuronales son notoriamente **sobreconfiadas**: tienden a producir probabilidades
extremas (>95%, <5%) que no se corresponden con la frecuencia real observada en el conjunto
de validación. Por ejemplo, antes de calibrar la MLP predicía probabilidad 80% para
pacientes cuya frecuencia real de cáncer era solo del 60%.

**Solución adoptada:** calibración isotónica post-hoc. Se entrena una función monótona
no decreciente sobre el conjunto de **validación** (no test, para evitar data leakage)
que mapea la probabilidad cruda del modelo a la probabilidad real observada.

**Justificación de elegir isotónica frente a Platt scaling:** la curva de descalibración
no era simétrica (sobreconfianza solo en la zona alta), por lo que la regresión isotónica
(no paramétrica, sin asunción de forma) era preferible al Platt scaling (sigmoide).

**Resultados (mejora del Brier Score sobre test):**

- Logistic Regression: 0.1689 → 0.1154 (+31.6%)
- Random Forest: 0.1619 → 0.1169 (+27.8%)
- XGBoost: 0.1440 → 0.1214 (+15.7%)
- LightGBM: 0.1485 → 0.1198 (+19.3%)
- **MLP: 0.1654 → 0.1170 (+29.3%)**

El AUC-ROC se mantiene prácticamente idéntico tras calibrar (0.8213 → 0.8212), confirmando
que la calibración no degrada la capacidad de discriminación, solo redistribuye las
probabilidades para que sean estadísticamente honestas.
""")

st.write("")


# ============================================================
#  CAP COSMÉTICO
# ============================================================
section_title("Cap visual de probabilidades extremas")

st.markdown("""
Aunque la calibración resuelve el problema estadístico, en la página de Evaluación de
Paciente se aplica un **cap visual adicional al rango [2%, 98%]** sobre las probabilidades
mostradas al usuario. Esta decisión sigue las **convenciones del software clínico real**
(Epic, Cerner) y responde a tres consideraciones:

1. **Responsabilidad legal:** ningún sistema de apoyo a la decisión clínica puede transmitir
   certeza absoluta (100% o 0%) sin invadir el rol diagnóstico del profesional sanitario.

2. **Comunicación con el usuario:** un médico desconfía instintivamente de un sistema que
   afirma "100% cáncer". Las probabilidades capadas mantienen la utilidad informativa
   sin generar falsa precisión.

3. **Honestidad epistémica:** los datos del estudio son sintéticos y limpios; en datos
   reales (con ruido, valores faltantes, comorbilidades imprevistas), las probabilidades
   extremas serían menos fiables.

**Importante:** el cap solo afecta a la presentación al usuario. Internamente, el sistema
sigue trabajando con las probabilidades calibradas sin capar para todas las métricas
y decisiones algorítmicas.
""")

st.write("")


# ============================================================
#  STACK TECNOLÓGICO
# ============================================================
section_title("Stack tecnológico")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="section-block" style="margin-top: 0;">
        <div style="font-size: 0.85rem; font-weight: 600; color: {COLORS['primary']};
                    margin-bottom: 8px;">Datos y procesamiento</div>
        <div style="font-size: 0.85rem; line-height: 1.7; color: {COLORS['text']};">
            Python 3.12<br>
            pandas 3.0<br>
            numpy 2.4<br>
            scikit-learn 1.8<br>
            pyarrow (parquet)
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="section-block" style="margin-top: 0;">
        <div style="font-size: 0.85rem; font-weight: 600; color: {COLORS['primary']};
                    margin-bottom: 8px;">Modelado</div>
        <div style="font-size: 0.85rem; line-height: 1.7; color: {COLORS['text']};">
            tensorflow 2.21 / keras 3.14<br>
            xgboost 3.2<br>
            lightgbm 4.6<br>
            imbalanced-learn 0.14
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="section-block" style="margin-top: 0;">
        <div style="font-size: 0.85rem; font-weight: 600; color: {COLORS['primary']};
                    margin-bottom: 8px;">Visualización y entrega</div>
        <div style="font-size: 0.85rem; line-height: 1.7; color: {COLORS['text']};">
            streamlit (interfaz)<br>
            plotly (gráficos)<br>
            matplotlib + seaborn<br>
            joblib (persistencia)
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ============================================================
#  LIMITACIONES Y ADVERTENCIAS
# ============================================================
section_title("Limitaciones y advertencias")

st.markdown("""
Esta sección documenta las **limitaciones conocidas** del sistema. Son fundamentales
para interpretar adecuadamente los resultados y orientan el uso responsable de la
herramienta.

**Sobre el dataset:**

- **Dataset sintético.** Los datos provienen de un modelo generativo basado en una
  combinación lineal de features con ruido gaussiano `ε ~ N(0, 0.8)`. Aunque la prevalencia
  y los pesos de los factores son realistas, los pacientes no corresponden a casos reales
  y no se han contrastado con cohortes clínicas externas.

- **Cobertura limitada.** El modelo solo evalúa 21 features predefinidas. Variables
  potencialmente relevantes en la práctica clínica (antecedentes familiares detallados,
  marcadores tumorales serológicos, imágenes diagnósticas, biopsias) no están presentes.

**Sobre el modelado:**

- **La calibración cambia el ranking.** En la comparativa sin calibrar (Fase 5),
  Random Forest lideraba con F1=0.5439. Tras la calibración isotónica (Fase 6),
  la MLP mejora a F1=0.5491 y pasa a ser el mejor modelo. La calibración no es un paso
  opcional: es parte del pipeline en producción.

- **Problema esencialmente lineal.** El AUC-ROC más alto lo obtiene la Logistic
  Regression (0.8275), lo que sugiere que las relaciones entre features y target son
  fundamentalmente lineales.

**Sobre el uso clínico:**

- **No es un sistema diagnóstico.** Esta herramienta proporciona una probabilidad
  estimada de riesgo oncológico. La decisión diagnóstica corresponde exclusivamente
  al profesional sanitario sobre la base de la historia clínica completa, exploración
  física y pruebas complementarias.

- **No sustituye el cribado oficial.** Los protocolos de cribado oncológico vigentes
  no se sustituyen por esta herramienta. El sistema puede actuar como complemento
  informativo, no como reemplazo.

- **Actualización requerida.** Cualquier despliegue clínico real requeriría re-entrenamiento
  con datos del propio centro, validación externa con poblaciones diversas y monitorización
  continua del rendimiento.
""")

st.write("")
medical_disclaimer()


# ============================================================
#  PIE DE PÁGINA
# ============================================================
st.markdown(f"""
<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid {COLORS['border']};
            font-size: 0.78rem; color: {COLORS['text_muted']}; text-align: center;">
Sistema de Evaluación de Riesgo Oncológico · Universidad Alfonso X el Sabio<br>
Asignatura de Inteligencia Artificial · Ingeniería Matemática · Curso 2025/2026<br>
Desarrollado por Álvaro Santamaría Antón
</div>
""", unsafe_allow_html=True)