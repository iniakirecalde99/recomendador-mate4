"""Modelo con sesgos por usuario y por película (rama experimento/sesgos).

r̂ᵢⱼ = μ + bᵢ + cⱼ + Uᵢ·Vⱼ, con μ fijo (promedio de las calificaciones de
entrenamiento del modelo) y bᵢ, cⱼ, U, V entrenados con ALS. Los pasos usan
`als.resolver_factor` sin modificarla, con la matriz aumentada con una
columna de unos. Esta rama no usa descenso de gradiente.
"""

import numpy as np

from src.modelo import predecir, validar_lambda


def calcular_mu(R: np.ndarray, M: np.ndarray) -> float:
    """μ: promedio de las calificaciones observadas en M.

    M es la máscara de entrenamiento del modelo: Ω_ent en la partición, todo
    Ω en el modelo reentrenado.
    """
    return float(np.mean(R[M]))


def predecir_con_sesgos(
    U: np.ndarray, V: np.ndarray, b: np.ndarray, c: np.ndarray, mu: float
) -> np.ndarray:
    """Estimación con sesgos r̂ᵢⱼ = μ + bᵢ + cⱼ + Uᵢ·Vⱼ."""
    # μ + sesgo del usuario + sesgo de la película + parte personal.
    return mu + b[:, None] + c[None, :] + predecir(U, V)


def f_sesgos(
    R: np.ndarray,
    M: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    mu: float,
    lambda_: float,
) -> float:
    """Función a minimizar con sesgos.

    f = Σ_{(i,j)∈Ω} (rᵢⱼ − r̂ᵢⱼ)² + λ (Σᵢ ‖Uᵢ‖² + Σⱼ ‖Vⱼ‖² + Σᵢ bᵢ² + Σⱼ cⱼ²),
    con r̂ de `predecir_con_sesgos`. Solo suma sobre M (los valores de R fuera
    de M no intervienen). Con μ, b y c en 0 es exactamente `f_regularizada`.
    Lanza `LambdaNegativoError` si λ < 0.
    """
    validar_lambda(lambda_)
    error = np.where(M, R - predecir_con_sesgos(U, V, b, c, mu), 0.0)
    penalizacion = float(np.sum(U**2) + np.sum(V**2) + np.sum(b**2) + np.sum(c**2))
    return float(np.sum(error**2)) + lambda_ * penalizacion
