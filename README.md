# Predicción de Riesgo Oncológico con Machine Learning y Redes Neuronales

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Caso práctico optativo · Asignatura de Inteligencia Artificial · UAX 2025/2026**  
**Autor: Álvaro Santamaría Antón**

---

Pipeline de Machine Learning completo para predecir riesgo oncológico sobre un dataset sintético de 50.001 pacientes. Incluye exploración de datos, preprocesado, 4 modelos clásicos de ML, una red neuronal multicapa (MLP), calibración isotónica post-hoc y un dashboard clínico interactivo en Streamlit.

---

## Resultados principales

| Modelo | F1-Score (test) | AUC-ROC | Threshold |
|---|---|---|---|
| **Random Forest** | **0.5439** | 0.8255 | 0.50 |
| MLP (calibrada) | 0.5491 | 0.8212 | 0.26 |
| Logistic Regression | 0.5385 | 0.8275 | 0.50 |
| LightGBM | 0.5383 | 0.8156 | 0.50 |
| XGBoost | 0.5207 | 0.8058 | 0.50 |

La MLP es el **modelo principal del sistema** por mayor estabilidad val→test, decisiones equilibradas y capacidad de ajuste del umbral. Todos los modelos incorporan calibración isotónica que mejora el Brier Score entre un 15% y un 32%.

---

## Qué contiene este repositorio

El repo contiene **el código fuente completo**. Los datos, modelos entrenados y artefactos procesados no se incluyen (ver `.gitignore`) y se regeneran ejecutando los notebooks en orden.

```
proyecto-cancer-ia/
│
├── src/                            # Módulos Python del pipeline
│   ├── data_loader.py              # Carga y merge de los 6 CSVs
│   ├── preprocessing.py            # Selección de features, splits, scaler
│   ├── models_ml.py                # Definición de los 4 modelos clásicos
│   ├── model_mlp.py                # Arquitectura de la MLP (Keras)
│   ├── evaluation.py               # Métricas, curvas ROC, matrices de confusión
│   └── inference.py                # Motor de inferencia para evaluación individual
│
├── notebooks/
│   ├── 01_eda.ipynb                # Análisis exploratorio
│   ├── 02_preprocessing.ipynb      # Preprocesado y validación
│   ├── 03_modelos_ml.ipynb         # Modelos clásicos de ML
│   ├── 04_mlp.ipynb                # Red neuronal multicapa
│   ├── 05_comparativa_final.ipynb  # Comparativa final en test
│   └── 06_auditoria_modelo.ipynb   # Calibración isotónica y auditoría
│
├── streamlit_app/
│   ├── .streamlit/config.toml      # Tema claro forzado
│   ├── assets/
│   │   ├── styles.css              # Hoja de estilos clínica
│   │   └── ui.py                   # Componentes visuales reutilizables
│   ├── pages/
│   │   ├── 1_Evaluacion_Paciente.py
│   │   ├── 2_Comparativa_Modelos.py
│   │   ├── 3_Analisis_Datos.py
│   │   └── 4_Metodologia.py
│   └── Inicio.py                   # Entry point del dashboard
│
├── reports/
│   └── comparativa_final_test.csv  # Tabla del ranking final (generada por notebook 05)
│
├── docs/
│   └── eda_summary.json            # Resumen del EDA (generado por notebook 01)
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Lo que no está en el repo (se genera localmente)

| Carpeta | Contenido | Se genera en |
|---|---|---|
| `data/raw/` | CSVs originales del dataset | Proporcionados por el docente |
| `data/processed/` | Splits `.parquet` y metadatos | Notebook 02 |
| `models/` | Modelos `.keras`, `.joblib`, calibradores | Notebooks 03, 04, 06 |
| `reports/figures/` | Figuras de los notebooks | Notebooks 01–06 |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/AlvaroSantamariaAnton/proyecto-cancer-ia.git
cd proyecto-cancer-ia
```

### 2. Crear el entorno virtual e instalar dependencias

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Colocar los CSVs originales

Copiar los 6 ficheros `CASOCANCER_0X_*.csv` en `data/raw/`.

---

## Reproducir el pipeline

Los notebooks deben ejecutarse en orden. Cada uno genera los artefactos que usa el siguiente.

```
01_eda.ipynb               → docs/eda_summary.json + 8 figuras
02_preprocessing.ipynb     → data/processed/ (splits + scaler)
03_modelos_ml.ipynb        → models/ (4 modelos clásicos)
04_mlp.ipynb               → models/mlp_cancer.keras + mlp_results_val.joblib
05_comparativa_final.ipynb → reports/comparativa_final_test.csv + figuras 19-22
06_auditoria_modelo.ipynb  → models/mlp_calibrator.joblib + ml_calibrators.joblib
```

Tiempo estimado: ~10 minutos (la mayor parte en el notebook 04).

---

## Lanzar el dashboard

```bash
cd streamlit_app
streamlit run Inicio.py
```

La app abre en `http://localhost:8501` con tema claro forzado.

### Páginas del dashboard

| Página | Descripción |
|---|---|
| **Inicio** | Panel ejecutivo con KPIs y resumen del sistema |
| **Evaluación de paciente** | Formulario clínico → predicción de riesgo con los 5 modelos |
| **Comparativa de modelos** | Métricas, curvas ROC y PR, matrices de confusión, calibración |
| **Análisis del dataset** | EDA, variables incluidas/excluidas, correlaciones, distribuciones |
| **Metodología** | Pipeline, arquitectura MLP, decisiones técnicas, limitaciones |

---

## Dataset

Dataset sintético generado mediante una función logística con ruido gaussiano (`ε ~ N(0, 0.8)`), proporcionado como parte del caso práctico.

| Característica | Valor |
|---|---|
| Pacientes | 50.001 |
| Ficheros origen | 6 CSVs |
| Features tras EDA | 21 |
| Prevalencia oncológica | 19.29% |
| Ratio de desbalance | 4.18 : 1 |

### Features del modelo (21)

**Bioquímica (7):** glucosa, colesterol, triglicéridos, hemoglobina, leucocitos, plaquetas, creatinina

**Marcadores genéticos (7):** mut_BRCA1, mut_TP53, mut_EGFR, mut_KRAS, mut_PIK3CA, mut_ALK, mut_BRAF

**Comorbilidades (4):** diabetes, hipertensión, obesidad, EPOC

**Estilo de vida y demografía (3):** fumador, actividad física (ordinal 0–2), edad

---

## Arquitectura de la MLP

```
Input (21)
   ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.25)
   ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.25)
   ↓
Dense(64)  → BatchNorm → ReLU → Dropout(0.20)
   ↓
Dense(1)   → Sigmoid
```

| Parámetro | Valor |
|---|---|
| Parámetros totales | 48.641 |
| Optimizador | Adam (lr=1e-3) |
| Épocas reales | 42 (EarlyStopping patience=12) |
| Mejor época | 30 |

---

## Calibración isotónica

| Modelo | Brier crudo | Brier calibrado | Mejora |
|---|---|---|---|
| Logistic Regression | 0.1689 | 0.1154 | +31.6% |
| Random Forest | 0.1619 | 0.1169 | +27.8% |
| XGBoost | 0.1440 | 0.1214 | +15.7% |
| LightGBM | 0.1485 | 0.1198 | +19.3% |
| MLP | 0.1654 | 0.1170 | +29.3% |

El AUC-ROC se mantiene prácticamente idéntico tras calibrar (Δ < 0.0002).

---

## Stack tecnológico

| Categoría | Librerías |
|---|---|
| Datos | pandas 3.0, numpy 2.4, pyarrow |
| Modelado | scikit-learn 1.8, tensorflow 2.21 / keras 3.14, xgboost 3.2, lightgbm 4.6 |
| Visualización | matplotlib 3.10, seaborn 0.13, plotly |
| Interfaz | streamlit |
| Persistencia | joblib |

---

## Advertencia

Este sistema es una herramienta de apoyo a la decisión clínica basada en datos sintéticos. No sustituye al juicio clínico ni a las pruebas diagnósticas. Cualquier uso en entorno real requeriría validación externa con datos clínicos reales.

---

## Licencia

Proyecto académico. Uso educativo.