"""Modelo con sesgos por usuario y por película (rama experimento/sesgos).

r̂ᵢⱼ = μ + bᵢ + cⱼ + Uᵢ·Vⱼ, con μ fijo (promedio de las calificaciones de
entrenamiento del modelo) y bᵢ, cⱼ, U, V entrenados con ALS. Los pasos usan
`als.resolver_factor` sin modificarla, con la matriz aumentada con una
columna de unos. Esta rama no usa descenso de gradiente.
"""

import numpy as np

from src.als import resolver_factor
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


def puntajes_ordenamiento_a(
    U: np.ndarray, V: np.ndarray, b: np.ndarray, c: np.ndarray, mu: float
) -> np.ndarray:
    """Ordenamiento A: por r̂ completo, μ + bᵢ + cⱼ + Uᵢ·Vⱼ."""
    return predecir_con_sesgos(U, V, b, c, mu)


def puntajes_ordenamiento_b(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Ordenamiento B: por la parte personal, Uᵢ·Vⱼ.

    μ y bᵢ no cambian el orden de las películas de un usuario; se saca cⱼ,
    así que B no depende del sesgo de las películas.
    """
    return predecir(U, V)


def _paso_aumentado(
    objetivo: np.ndarray, M: np.ndarray, F: np.ndarray, lambda_: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resuelve un paso de ALS con sesgo: incógnitas (factor, sesgo) ∈ R^(k+1).

    Aumenta F con una columna de unos, [F | 1], y llama a `resolver_factor`
    sin modificarla: por fila resuelve ([F | 1]ᵀ [F | 1] + λ I) x = [F | 1]ᵀ y,
    que regulariza también el sesgo (el término λ·sesgo² de la f).
    Devuelve (factor de k columnas, sesgo).
    """
    F_aumentada = np.hstack([F, np.ones((F.shape[0], 1))])
    solucion = resolver_factor(objetivo, M, F_aumentada, lambda_=lambda_)
    return solucion[:, :-1], solucion[:, -1]


def paso_usuarios(
    R: np.ndarray, M: np.ndarray, V: np.ndarray, c: np.ndarray, mu: float, lambda_: float
) -> tuple[np.ndarray, np.ndarray]:
    """Paso de usuarios: (Uᵢ, bᵢ) con [V | 1] y objetivo rᵢⱼ − μ − cⱼ.

    También calcula el vector (u, b) de un usuario simulado. Los
    NaN de R siguen siendo NaN en el objetivo; solo cuenta M.
    """
    # Paso de usuarios con la matriz aumentada [V | 1].
    return _paso_aumentado(R - mu - c[None, :], M, V, lambda_)


def paso_peliculas(
    R: np.ndarray, M: np.ndarray, U: np.ndarray, b: np.ndarray, mu: float, lambda_: float
) -> tuple[np.ndarray, np.ndarray]:
    """Paso de películas: (Vⱼ, cⱼ) con [U | 1] y objetivo rᵢⱼ − μ − bᵢ.

    Es el mismo paso aumentado que el de usuarios, llamado con la matriz
    transpuesta.
    """
    # Paso de películas, misma función con (R − μ − b)ᵀ, Mᵀ y [U | 1].
    return _paso_aumentado((R - mu - b[:, None]).T, M.T, U, lambda_)
