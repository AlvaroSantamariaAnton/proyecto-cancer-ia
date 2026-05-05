"""
src/inference.py
================
Módulo de inferencia para predicción de cáncer de un paciente individual.

Carga los 5 modelos entrenados (4 ML + MLP) y proporciona funciones para:
  - Convertir datos crudos de un paciente a vector de features escalado
  - Predecir riesgo de cáncer con cada modelo
  - Calcular factores de riesgo destacados respecto a poblaciones de referencia
  - Generar contexto clínico para visualización
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

# Lazy imports de Keras: solo cuando se necesita
_keras = None


def _load_keras():
    global _keras
    if _keras is None:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import keras
        _keras = keras
    return _keras


# === Constantes y rutas ===

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Orden de las 21 features (debe coincidir EXACTAMENTE con el del entrenamiento)
FEATURE_ORDER = [
    "glucosa", "colesterol", "trigliceridos", "hemoglobina",
    "leucocitos", "plaquetas", "creatinina",
    "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
    "mut_PIK3CA", "mut_ALK", "mut_BRAF",
    "diabetes", "hipertension", "obesidad", "epoc",
    "fumador", "actividad_fisica", "edad",
]

MAPEO_ACTIVIDAD = {"Baja": 0, "Moderada": 1, "Alta": 2}

# Rangos de referencia clínica (para detección de valores anómalos)
# Fuente: rangos clínicos estándar + metadata del dataset
RANGOS_NORMALES = {
    "glucosa":      {"min": 70,  "max": 100, "unidad": "mg/dL", "umbral_alto": 126},
    "colesterol":   {"min": 0,   "max": 200, "unidad": "mg/dL", "umbral_alto": 240},
    "trigliceridos":{"min": 0,   "max": 150, "unidad": "mg/dL", "umbral_alto": 200},
    "hemoglobina":  {"min": 12,  "max": 16,  "unidad": "g/dL",  "umbral_bajo": 11},
    "leucocitos":   {"min": 4.5, "max": 11,  "unidad": "×10³/µL","umbral_alto": 11},
    "plaquetas":    {"min": 150, "max": 400, "unidad": "×10³/µL","umbral_alto": 400, "umbral_bajo": 150},
    "creatinina":   {"min": 0.6, "max": 1.3, "unidad": "mg/dL", "umbral_alto": 1.3},
}

# Pesos del modelo generativo (de la metadata) — útil para explicar el riesgo
PESOS_FACTORES = {
    "mut_BRCA1": 2.0, "mut_TP53": 1.8, "fumador": 1.5,
    "mut_KRAS": 1.4,  "glucosa_alta": 1.2, "obesidad": 1.1,
    "mut_EGFR": 1.0,  "hemoglobina_baja": 0.9, "mut_PIK3CA": 0.8,
    "leucocitos_altos": 0.7, "mut_BRAF": 0.6, "hipertension": 0.5,
    "edad_alta": 0.4, "actividad_alta": -1.2, "actividad_moderada": -0.6,
}


# === Carga de artefactos ===

class InferenceEngine:
    """
    Motor de inferencia que carga todos los modelos y la información
    necesaria para evaluar pacientes individuales.
    """

    def __init__(self):
        self.scaler = None
        self.models_ml = {}
        self.model_mlp = None
        self.mlp_threshold = 0.5
        self.feature_order = FEATURE_ORDER
        self.population_stats = None
        self.mlp_calibrator = None
        self.ml_calibrators = {}        # ← NUEVO
        self.calibrated = False
        self._loaded = False

    def load(self) -> "InferenceEngine":
        """Carga el scaler, los 4 modelos ML, la MLP, el calibrador y las estadísticas poblacionales."""
        if self._loaded:
            return self

        # 1. Scaler
        self.scaler = joblib.load(MODELS_DIR / "scaler.joblib")

        # 2. Modelos clásicos
        self.models_ml = {
            "Logistic Regression": joblib.load(MODELS_DIR / "logistic_regression.joblib"),
            "Random Forest":       joblib.load(MODELS_DIR / "random_forest.joblib"),
            "XGBoost":              joblib.load(MODELS_DIR / "xgboost.joblib"),
            "LightGBM":             joblib.load(MODELS_DIR / "lightgbm.joblib"),
        }

        # 3. MLP
        keras = _load_keras()
        self.model_mlp = keras.models.load_model(MODELS_DIR / "mlp_cancer.keras")

        # 4. Calibrador isotónico (nuevo)
        # Aplica calibración post-hoc a las probabilidades crudas de la MLP.
        # Justificación en notebook 06_auditoria_modelo.ipynb:
        #   - La MLP cruda tiene Brier 0.1654; calibrada baja a 0.1170 (+29% mejor).
        #   - AUC se mantiene idéntico, el F1 mejora ligeramente.
        #   - Threshold óptimo bajó de 0.68 (crudo) a 0.26 (calibrado).
        try:
            calibrator_data = joblib.load(MODELS_DIR / "mlp_calibrator.joblib")
            self.mlp_calibrator = calibrator_data["calibrator"]
            self.mlp_threshold = calibrator_data["threshold_calibrated"]
            self.calibrated = True
        except FileNotFoundError:
            # Fallback al threshold sin calibrar si no hay calibrador
            mlp_results = joblib.load(MODELS_DIR / "mlp_results_val.joblib")
            self.mlp_calibrator = None
            self.mlp_threshold = mlp_results["threshold_optimal"]
            self.calibrated = False

        # 4b. Calibradores isotónicos de los 4 modelos ML
        try:
            self.ml_calibrators = joblib.load(MODELS_DIR / "ml_calibrators.joblib")
        except FileNotFoundError:
            self.ml_calibrators = {}

        # 5. Estadísticas poblacionales (para comparar al paciente con grupos de referencia)
        self.population_stats = self._compute_population_stats()

        self._loaded = True
        return self

    def _compute_population_stats(self) -> dict:
        """Calcula medias y std de cada feature, separando por clase (cancer 0/1).
        Sirve para mostrar 'el paciente está cerca del grupo X'."""
        X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
        y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["cancer"]

        # IMPORTANTE: X_train está YA escalado. Para comparar con valores crudos
        # del paciente, necesitamos volver a las unidades originales.
        # Inversa del escalado: x = z * std + mean
        scaler_means = self.scaler.mean_
        scaler_scales = self.scaler.scale_

        X_train_raw = X_train.values * scaler_scales + scaler_means
        df_raw = pd.DataFrame(X_train_raw, columns=self.feature_order)
        df_raw["cancer"] = y_train.values

        stats = {}
        for feat in self.feature_order:
            stats[feat] = {
                "media_global":  float(df_raw[feat].mean()),
                "media_no_cancer": float(df_raw.loc[df_raw["cancer"] == 0, feat].mean()),
                "media_cancer":    float(df_raw.loc[df_raw["cancer"] == 1, feat].mean()),
                "std_global":    float(df_raw[feat].std()),
            }
        return stats


# === Predicción de un paciente ===

def patient_dict_to_vector(patient: dict, feature_order: list) -> np.ndarray:
    """
    Convierte un diccionario con datos crudos del paciente en un vector ordenado.

    Parameters
    ----------
    patient : dict
        Diccionario con todas las features. Para 'actividad_fisica' acepta
        tanto el string ("Baja"/"Moderada"/"Alta") como ya el ordinal (0/1/2).
    feature_order : list
        Orden esperado de las features.

    Returns
    -------
    np.ndarray de shape (1, n_features) listo para escalar.
    """
    row = []
    for feat in feature_order:
        value = patient[feat]
        if feat == "actividad_fisica" and isinstance(value, str):
            value = MAPEO_ACTIVIDAD[value]
        row.append(float(value))
    return np.array(row).reshape(1, -1)


def predict_patient(engine: InferenceEngine, patient: dict) -> dict:
    """
    Predice riesgo de cáncer para un paciente individual con los 5 modelos.

    Parameters
    ----------
    engine : InferenceEngine
        Motor cargado con load().
    patient : dict
        Datos crudos del paciente (21 features).

    Returns
    -------
    dict con:
        - probabilities  : {nombre_modelo: probabilidad [0-1]}
        - predictions    : {nombre_modelo: 0 o 1}
        - thresholds     : {nombre_modelo: threshold usado}
        - mlp_probability: probabilidad de la MLP (modelo recomendado)
        - mlp_prediction : 0 o 1 con threshold óptimo
        - risk_level     : 'Alto' / 'Moderado' / 'Bajo'
        - patient_raw    : el diccionario de entrada
        - patient_vector : array escalado usado para predicción
    """
    if not engine._loaded:
        raise RuntimeError("Engine no cargado. Llama a engine.load() primero.")

    # 1. Construir vector y escalarlo
    x_raw = patient_dict_to_vector(patient, engine.feature_order)
    x_scaled = engine.scaler.transform(x_raw)

    # 2. Predicciones con los 4 modelos clásicos
    #    Probabilidad CRUDA → calibración isotónica si hay calibrador → threshold 0.5
    probabilities = {}
    predictions = {}
    thresholds = {}
    for name, model in engine.models_ml.items():
        prob_raw = float(model.predict_proba(x_scaled)[0, 1])
        # Calibrar si hay calibrador disponible
        if name in engine.ml_calibrators:
            prob = float(engine.ml_calibrators[name].predict([prob_raw])[0])
        else:
            prob = prob_raw
        probabilities[name] = prob
        predictions[name] = int(prob >= 0.5)
        thresholds[name] = 0.5

    # 3. Predicción con la MLP: tres niveles de probabilidad
    #    raw      → probabilidad bruta de la red (sobreconfiada por construcción)
    #    cal      → probabilidad estadísticamente correcta (calibrada en val)
    #    display  → probabilidad para mostrar al usuario, capada a [0.02, 0.98]
    #               por convención de software clínico
    prob_mlp_raw = float(engine.model_mlp.predict(x_scaled, verbose=0).ravel()[0])

    # Aplicar calibración isotónica
    if engine.calibrated and engine.mlp_calibrator is not None:
        prob_mlp_cal = float(engine.mlp_calibrator.predict([prob_mlp_raw])[0])
    else:
        prob_mlp_cal = prob_mlp_raw

    # Cap cosmético: [0.02, 0.98]. Razones:
    #   - Ningún test médico tiene certeza del 100%
    #   - Convención de software clínico (Epic, Cerner)
    #   - Evita transmitir "diagnóstico" en lugar de "apoyo a la decisión"
    prob_mlp_display = max(0.02, min(0.98, prob_mlp_cal))

    # Para el resto del código (predicciones, threshold), usamos la calibrada SIN capar
    # para que el resultado del threshold sea consistente con la métrica F1 calculada.
    prob_mlp = prob_mlp_cal

    name_mlp = f"MLP (umbral óptimo)"
    probabilities[name_mlp] = prob_mlp
    predictions[name_mlp] = int(prob_mlp >= engine.mlp_threshold)
    thresholds[name_mlp] = engine.mlp_threshold

    # 4. Nivel de riesgo basado en la probabilidad CALIBRADA de la MLP
    # Con calibración, las probabilidades reflejan la frecuencia real de cáncer:
    #   >=0.50 = más probable que sea cáncer que no (clínicamente alto)
    #   >=0.26 = umbral de decisión óptimo (moderado-alto)
    #   <0.10 = riesgo claramente por debajo de la prevalencia base (19.3%)
    if prob_mlp >= 0.50:
        risk_level = "Alto"
    elif prob_mlp >= engine.mlp_threshold:   # 0.26
        risk_level = "Moderado"
    else:
        risk_level = "Bajo"

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "thresholds": thresholds,
        "mlp_probability": prob_mlp,                  # Calibrada SIN capar (para lógica del modelo)
        "mlp_probability_display": prob_mlp_display,  # Capada [0.02, 0.98] (para mostrar al usuario)
        "mlp_probability_raw": prob_mlp_raw,          # Cruda (solo para diagnóstico técnico)
        "mlp_calibrated": engine.calibrated,
        "mlp_prediction": predictions[name_mlp],
        "mlp_threshold": engine.mlp_threshold,
        "risk_level": risk_level,
        "patient_raw": patient,
        "patient_vector_scaled": x_scaled,
    }


def detect_risk_factors(patient: dict) -> list:
    """
    Devuelve una lista de factores de riesgo presentes en el paciente,
    ordenados por su peso en el modelo generativo (de mayor a menor).

    Returns
    -------
    list[dict] con campos: factor, descripcion, severidad, peso
    """
    factores = []

    # === Mutaciones genéticas ===
    mutaciones_info = [
        ("mut_BRCA1", "Mutación BRCA1 detectada", 2.0, "alta"),
        ("mut_TP53",  "Mutación TP53 detectada",  1.8, "alta"),
        ("mut_KRAS",  "Mutación KRAS detectada",  1.4, "alta"),
        ("mut_EGFR",  "Mutación EGFR detectada",  1.0, "media"),
        ("mut_PIK3CA","Mutación PIK3CA detectada",0.8, "media"),
        ("mut_BRAF",  "Mutación BRAF detectada",  0.6, "baja"),
        ("mut_ALK",   "Mutación ALK detectada (sin peso predictivo en este modelo)", 0.0, "info"),
    ]
    for key, desc, peso, sev in mutaciones_info:
        if patient.get(key, 0) == 1:
            factores.append({"factor": key, "descripcion": desc, "peso": peso, "severidad": sev})

    # === Hábitos ===
    if patient.get("fumador", 0) == 1:
        factores.append({"factor": "fumador",
                         "descripcion": "Tabaquismo activo",
                         "peso": 1.5, "severidad": "alta"})

    af = patient.get("actividad_fisica", 0)
    if isinstance(af, str):
        af = MAPEO_ACTIVIDAD.get(af, 0)
    if af == 0:
        factores.append({"factor": "actividad_baja",
                         "descripcion": "Actividad física baja (sin protección)",
                         "peso": 0.0, "severidad": "media"})
    elif af == 2:
        factores.append({"factor": "actividad_alta",
                         "descripcion": "Actividad física alta (factor protector)",
                         "peso": -1.2, "severidad": "protector"})

    # === Bioquímica ===
    if patient.get("glucosa", 0) > 130:
        factores.append({"factor": "glucosa_alta",
                         "descripcion": f"Glucosa elevada ({patient['glucosa']:.0f} mg/dL, >130)",
                         "peso": 1.2, "severidad": "alta"})
    elif patient.get("glucosa", 0) > 100:
        factores.append({"factor": "glucosa_borderline",
                         "descripcion": f"Glucosa en límite alto ({patient['glucosa']:.0f} mg/dL)",
                         "peso": 0.0, "severidad": "info"})

    if patient.get("hemoglobina", 99) < 11:
        factores.append({"factor": "hemoglobina_baja",
                         "descripcion": f"Hemoglobina baja ({patient['hemoglobina']:.1f} g/dL, <11)",
                         "peso": 0.9, "severidad": "alta"})

    if patient.get("leucocitos", 0) > 10:
        factores.append({"factor": "leucocitos_altos",
                         "descripcion": f"Leucocitos elevados ({patient['leucocitos']:.1f} ×10³/µL, >10)",
                         "peso": 0.7, "severidad": "media"})

    # === Comorbilidades ===
    if patient.get("obesidad", 0) == 1:
        factores.append({"factor": "obesidad", "descripcion": "Obesidad (IMC ≥ 30)",
                         "peso": 1.1, "severidad": "alta"})
    if patient.get("hipertension", 0) == 1:
        factores.append({"factor": "hipertension", "descripcion": "Hipertensión arterial",
                         "peso": 0.5, "severidad": "media"})
    if patient.get("diabetes", 0) == 1:
        factores.append({"factor": "diabetes", "descripcion": "Diabetes mellitus",
                         "peso": 0.0, "severidad": "info"})
    if patient.get("epoc", 0) == 1:
        factores.append({"factor": "epoc", "descripcion": "EPOC diagnosticada",
                         "peso": 0.0, "severidad": "info"})

    # === Edad ===
    if patient.get("edad", 0) > 55:
        factores.append({"factor": "edad_alta",
                         "descripcion": f"Edad de riesgo ({patient['edad']} años, >55)",
                         "peso": 0.4, "severidad": "media"})

    # Ordenar por peso (factores protectores al final)
    factores.sort(key=lambda x: -x["peso"])
    return factores


# === Smoke test ===

if __name__ == "__main__":
    print("Probando inference.py...\n")

    engine = InferenceEngine().load()
    print(f"Engine cargado:")
    print(f"  Modelos clásicos: {list(engine.models_ml.keys())}")
    print(f"  MLP threshold:    {engine.mlp_threshold:.2f}")
    print(f"  Features:         {len(engine.feature_order)}")

    # Paciente de ejemplo: alto riesgo
    paciente_test = {
        "glucosa": 145, "colesterol": 230, "trigliceridos": 220,
        "hemoglobina": 10.5, "leucocitos": 11.5, "plaquetas": 280, "creatinina": 1.0,
        "mut_BRCA1": 1, "mut_TP53": 1, "mut_EGFR": 0, "mut_KRAS": 0,
        "mut_PIK3CA": 0, "mut_ALK": 0, "mut_BRAF": 0,
        "diabetes": 1, "hipertension": 1, "obesidad": 1, "epoc": 0,
        "fumador": 1, "actividad_fisica": "Baja", "edad": 62,
    }

    print(f"\n--- PACIENTE DE PRUEBA (alto riesgo) ---")
    result = predict_patient(engine, paciente_test)
    print(f"  Nivel de riesgo: {result['risk_level']}")
    print(f"  Probabilidad MLP cruda:     {result['mlp_probability_raw']:.4f}")
    print(f"  Probabilidad MLP calibrada: {result['mlp_probability']:.4f}")
    print(f"  Probabilidad MLP display:   {result['mlp_probability_display']:.4f}  ← lo que ve el usuario")
    print(f"  Threshold:                  {result['mlp_threshold']:.2f}")
    print(f"  Predicción MLP:             {'CÁNCER' if result['mlp_prediction'] == 1 else 'NO CÁNCER'}")
    print(f"\n  Probabilidades por modelo:")
    for name, p in result["probabilities"].items():
        bar = "█" * int(p * 30)
        print(f"    {name:25s} {p:.4f}  {bar}")

    factores = detect_risk_factors(paciente_test)
    print(f"\n  Factores de riesgo detectados: {len(factores)}")
    for f in factores[:5]:
        print(f"    [{f['severidad']:8s}] {f['descripcion']}  (peso={f['peso']:+.1f})")

    print("\nOK — inference.py funciona correctamente.")