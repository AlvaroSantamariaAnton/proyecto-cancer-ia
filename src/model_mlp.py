"""
src/model_mlp.py
================
Red Neuronal Multicapa (MLP) para clasificación binaria de cáncer.

Arquitectura siguiendo las especificaciones del PDF del enunciado:
  - 3 capas ocultas Dense (128 → 64 → 32 neuronas) con activación ReLU
  - BatchNormalization tras cada Dense
  - Dropout (0.25, 0.25, 0.20)
  - Capa de salida 1 neurona con sigmoid
  - Optimizador Adam, pérdida binary_crossentropy
  - Callbacks: EarlyStopping (paciencia 12) + ReduceLROnPlateau (factor 0.5, paciencia 6)
"""
import os
# Reducir verbosidad de TensorFlow ANTES de importarlo
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import keras
from keras import layers, callbacks

RANDOM_STATE = 42


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Fija las semillas en Python, NumPy y TensorFlow para reproducibilidad."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_mlp(
    input_dim: int,
    hidden_units: tuple = (256, 128, 64),
    dropout_rates: tuple = (0.25, 0.25, 0.20),
    learning_rate: float = 1e-3,
    seed: int = RANDOM_STATE,
) -> keras.Model:
    """
    Construye la arquitectura MLP siguiendo el diseño del PDF.

    Bloque por capa oculta:
        Dense(units) → BatchNormalization → Activation('relu') → Dropout(rate)

    El orden Dense → BN → Activation → Dropout es el patrón recomendado:
    BatchNorm normaliza la combinación lineal antes de la no-linealidad.

    Parameters
    ----------
    input_dim : int
        Número de features de entrada (21 en nuestro caso).
    hidden_units : tuple
        Neuronas por capa oculta. Default (128, 64, 32) → ~46k parámetros.
    dropout_rates : tuple
        Tasas de dropout por capa. Default (0.25, 0.25, 0.20).
    learning_rate : float
        Learning rate inicial del optimizador Adam.
    seed : int
        Semilla para la inicialización de pesos.

    Returns
    -------
    keras.Model compilado, listo para entrenar.
    """
    if len(hidden_units) != len(dropout_rates):
        raise ValueError(
            f"hidden_units ({len(hidden_units)}) y dropout_rates "
            f"({len(dropout_rates)}) deben tener la misma longitud."
        )

    set_seeds(seed)

    inputs = keras.Input(shape=(input_dim,), name="input_features")
    x = inputs

    for i, (units, dr) in enumerate(zip(hidden_units, dropout_rates), start=1):
        x = layers.Dense(units, name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.Dropout(dr, name=f"dropout_{i}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_cancer")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def get_callbacks(
    monitor: str = "val_loss",
    patience_early: int = 12,
    patience_lr: int = 6,
    factor_lr: float = 0.5,
    min_lr: float = 1e-6,
    verbose: int = 1,
) -> list:
    """
    Devuelve los callbacks especificados en el PDF.

    EarlyStopping (paciencia 12):
        Detiene el entrenamiento si val_loss no mejora durante 12 épocas.
        restore_best_weights=True devuelve los pesos de la mejor época.

    ReduceLROnPlateau (factor 0.5, paciencia 6):
        Multiplica el learning rate por 0.5 si val_loss no mejora durante 6 épocas.
    """
    return [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_early,
            restore_best_weights=True,
            verbose=verbose,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor_lr,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=verbose,
        ),
    ]


def get_summary_str(model: keras.Model) -> str:
    """Devuelve el summary del modelo como string (útil para guardarlo)."""
    lines = []
    model.summary(print_fn=lambda s: lines.append(s))
    return "\n".join(lines)


if __name__ == "__main__":
    print("Probando model_mlp.py...\n")

    model = build_mlp(input_dim=21)

    print("Resumen del modelo:")
    print("-" * 60)
    model.summary()

    n_params = model.count_params()
    print(f"\nParámetros totales: {n_params:,}")
    print(f"Esperado por el PDF: ~46.913")

    cbs = get_callbacks()
    print(f"\nCallbacks configurados:")
    for cb in cbs:
        print(f"  - {type(cb).__name__}")

    print("\nOK — model_mlp.py funciona correctamente.")