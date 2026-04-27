# Predicción de Diagnóstico de Cáncer

Caso de uso práctico de la asignatura **Inteligencia Artificial** (Ingeniería Matemática, UAX, curso 2025–2026).

## Objetivo

Evaluar la viabilidad de un sistema de cribado de cáncer mediante Machine Learning, comparando algoritmos clásicos con una Red Neuronal Multicapa (MLP) sobre un dataset multimodal de 50.001 pacientes sintéticos.

## Datos

Cinco colecciones CSV unidas por `paciente_id`:
- `CASOCANCER_01_BIOQUIMICOS.csv` — analíticas de sangre
- `CASOCANCER_02_CLINICOS.csv` — variables clínicas
- `CASOCANCER_03_GENETICOS.csv` — marcadores genéticos
- `CASOCANCER_05_GENERALES.csv` — datos generales del paciente
- `CASOCANCER_06_SOCIODEMOGRAFICOS.csv` — variables sociodemográficas

Los CSV no se incluyen en el repositorio. Deben descargarse del campus virtual y colocarse en `data/raw/`.

## Estructura del proyecto

```
proyecto-cancer-ia/
├── data/
│   ├── raw/              # CSVs originales (no en git)
│   └── processed/        # Dataset unificado y splits
├── notebooks/            # Jupyter notebooks (EDA, experimentos)
├── src/                  # Código modular .py reutilizable
├── models/               # Modelos entrenados (no en git)
├── reports/
│   ├── figures/          # Gráficos generados
│   └── slides/           # Las 5 diapositivas finales
├── docs/                 # Documentación adicional (metadata)
├── requirements.txt      # Dependencias del proyecto
└── README.md
```

## Setup del entorno

Requiere **Python 3.12** (TensorFlow 2.21 no soporta Python 3.14 todavía).

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Plan de trabajo

- [x] Fase 0 — Setup del entorno y repositorio
- [ ] Fase 1 — EDA (análisis exploratorio)
- [ ] Fase 2 — Preprocesado y construcción del dataset unificado
- [ ] Fase 3 — Modelos baseline + ML clásicos
- [ ] Fase 4 — Tuning de hiperparámetros
- [ ] Fase 5 — Diseño y entrenamiento de la MLP
- [ ] Fase 6 — Validación de la MLP (curvas)
- [ ] Fase 7 — Optimización del umbral de clasificación
- [ ] Fase 8 — Evaluación final y comparativa
- [ ] Fase 9 — 5 diapositivas

## Autor

Álvaro Santamaría Antón