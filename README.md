# Predicción de Riesgo Oncológico con Machine Learning y Redes Neuronales

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B?logo=streamlit&logoColor=white)](https://proyecto-cancer-ia-alvaro-santamaria.streamlit.app)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E)](https://scikit-learn.org/)

**Caso práctico optativo · Asignatura de Inteligencia Artificial · UAX 2025/2026**  
**Autor: Álvaro Santamaría Antón**

---

## Demo

[**Acceder al dashboard clínico →**](https://proyecto-cancer-ia-alvaro-santamaria.streamlit.app)

---

## Descripción

Pipeline de Machine Learning end-to-end para predecir riesgo oncológico sobre un dataset sintético de 50.001 pacientes. El proyecto incluye exploración de datos, preprocesado, 4 modelos clásicos de ML, una red neuronal multicapa (MLP), calibración isotónica post-hoc aplicada a los 5 modelos, y un dashboard clínico interactivo desplegado en Streamlit Cloud con apariencia de software hospitalario (estilo Epic/Cerner).

---

## Resultados principales

> Resultados con calibración isotónica post-hoc aplicada a los 5 modelos.

| # | Modelo | F1-Score | AUC-ROC | Precisión | Recall |
|---|---|---|---|---|---|
| **1** | **MLP calibrada (t=0.26)** | **0.5491** | 0.8212 | 0.5030 | 0.6045 |
| 2 | Random Forest | 0.5439 | 0.8255 | 0.4341 | 0.7278 |
| 3 | Logistic Regression | 0.5385 | **0.8275** | 0.4217 | 0.7449 |
| 4 | LightGBM | 0.5383 | 0.8156 | 0.4586 | 0.6516 |
| 5 | XGBoost | 0.5207 | 0.8058 | 0.4653 | 0.5910 |

Sin calibración, Random Forest lideraba con F1=0.5439 y la MLP era segunda con F1=0.5423. Tras aplicar calibración isotónica (Fase 6), la MLP mejora a F1=0.5491 y pasa a liderar. La calibración es parte del pipeline en producción, no un paso opcional.

---

## Qué contiene este repositorio

El repo contiene el **código fuente completo** y los **artefactos necesarios para ejecutar la app**. Los datos crudos (CSVs) y las figuras PNG de los notebooks no se incluyen.

```
proyecto-cancer-ia/
│
├── src/                            # Módulos Python del pipeline
│   ├── data_loader.py              # Carga y merge de los 6 CSVs
│   ├── preprocessing.py            # Selección de features, splits, scaler
│   ├── models_ml.py                # Definición de los 4 modelos clásicos
│   ├── model_mlp.py                # Arquitectura de la MLP (Keras)
│   ├── evaluation.py               # Métricas, curvas ROC, matrices de confusión
│   └── inference.py                # Motor de inferencia (calibración incluida)
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
│   └── Inicio.py
│
├── models/                         # Modelos y calibradores entrenados
│   ├── scaler.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── lightgbm.joblib
│   ├── mlp_cancer.keras
│   ├── mlp_calibrator.joblib       # Calibrador isotónico de la MLP
│   ├── ml_calibrators.joblib       # Calibradores de los 4 modelos clásicos
│   ├── mlp_history.joblib          # Historial de entrenamiento
│   ├── mlp_results_val.joblib      # Resultados en validación
│   └── results_final_test.joblib   # Resultados en test
│
├── data/processed/
│   ├── master_dataset.parquet      # Dataset mergeado (para análisis en app)
│   ├── y_val.parquet
│   └── y_test.parquet
│
├── docs/
│   └── eda_summary.json
│
├── reports/
│   └── comparativa_final_test.csv  # Ranking final con métricas calibradas
│
├── .python-version                 # Python 3.12 para Streamlit Cloud
├── requirements.txt                # Dependencias de la app
├── requirements-dev.txt            # Dependencias completas (notebooks + app)
└── README.md
```

---

## Instalación

```bash
git clone https://github.com/AlvaroSantamariaAnton/proyecto-cancer-ia.git
cd proyecto-cancer-ia

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements-dev.txt
```

---

## Reproducir el pipeline completo

Requiere los 6 CSVs originales en `data/raw/`. Ejecutar los notebooks en orden:

```
01_eda.ipynb               → docs/eda_summary.json
02_preprocessing.ipynb     → data/processed/ (splits + scaler)
03_modelos_ml.ipynb        → models/ (4 modelos clásicos)
04_mlp.ipynb               → models/mlp_cancer.keras + historial
05_comparativa_final.ipynb → reports/comparativa_final_test.csv
06_auditoria_modelo.ipynb  → models/mlp_calibrator.joblib + ml_calibrators.joblib
```

Tiempo estimado: ~10 minutos (la mayor parte en el notebook 04).

---

## Lanzar el dashboard en local

```bash
cd streamlit_app
streamlit run Inicio.py
```

La app abre en `http://localhost:8501` con tema claro forzado.

---

## Páginas del dashboard

| Página | Descripción |
|---|---|
| **Inicio** | Panel ejecutivo con KPIs del sistema |
| **Evaluación de paciente** | Formulario clínico → predicción con los 5 modelos calibrados, factores de riesgo, radar chart poblacional |
| **Comparativa de modelos** | Métricas, curvas ROC y PR, matrices de confusión, calibración antes/después, curvas de entrenamiento MLP |
| **Análisis del dataset** | Prevalencia, distribuciones, correlaciones, variables excluidas |
| **Metodología** | Pipeline completo, arquitectura MLP, decisiones técnicas, limitaciones |

---

## Dataset

Dataset sintético generado mediante función logística con ruido gaussiano (`ε ~ N(0, 0.8)`).

| Característica | Valor |
|---|---|
| Pacientes | 50.001 |
| Ficheros origen | 6 CSVs |
| Features tras EDA | 21 |
| Prevalencia oncológica | 19.29% |
| Ratio de desbalance | 4.18 : 1 |
| División | 60% train / 20% val / 20% test |

---

## Arquitectura de la MLP

```
Input (21)
   ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.25)    448 neuronas ocultas
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
| Neuronas ocultas | 448 (256 + 128 + 64) |
| Optimizador | Adam (lr=1e-3) |
| Épocas reales | 42 (EarlyStopping patience=12) |
| Mejor época | 30 |
| Threshold óptimo (crudo) | 0.68 |
| Threshold óptimo (calibrado) | 0.26 |

---

## Calibración isotónica post-hoc

Auditoría previa detectó sobreconfianza severa: cuando la MLP predecía 80%, la frecuencia real de cáncer era solo 60%. Se aplicó calibración isotónica entrenada en validación a los 5 modelos.

| Modelo | Brier crudo | Brier calibrado | Mejora |
|---|---|---|---|
| Logistic Regression | 0.1689 | 0.1154 | +31.6% |
| Random Forest | 0.1619 | 0.1169 | +27.8% |
| XGBoost | 0.1440 | 0.1214 | +15.7% |
| LightGBM | 0.1485 | 0.1198 | +19.3% |
| **MLP** | **0.1654** | **0.1170** | **+29.3%** |

El AUC-ROC se mantiene prácticamente idéntico tras calibrar (Δ < 0.0002). La app muestra probabilidades calibradas capadas visualmente a [2%, 98%] siguiendo las convenciones de software clínico real.

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

Este sistema es una herramienta de apoyo a la decisión clínica basada en datos sintéticos. No sustituye al juicio clínico ni a las pruebas diagnósticas confirmatorias. Cualquier uso en entorno real requeriría validación externa con datos clínicos reales, re-entrenamiento con datos del propio centro y monitorización continua del rendimiento.