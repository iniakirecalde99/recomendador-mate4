"""Modelo con sesgos por usuario y por película (rama experimento/sesgos).

r̂ᵢⱼ = μ + bᵢ + cⱼ + Uᵢ·Vⱼ, con μ fijo (promedio de las calificaciones de
entrenamiento del modelo) y bᵢ, cⱼ, U, V entrenados con ALS. Los pasos usan
`als.resolver_factor` sin modificarla, con la matriz aumentada con una
columna de unos. Esta rama no usa descenso de gradiente.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

from src.als import resolver_factor
from src.modelo import predecir, validar_lambda

LOGGER = logging.getLogger(__name__)


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


@dataclass
class ResultadoSesgos:
    """Resultado de entrenar ALS con sesgos.

    Atributos:
        U, V: factores finales.
        b, c: sesgos finales por usuario y por película.
        mu: el μ fijo con el que se entrenó.
        historial_f: valor de `f_sesgos` en cada iteración (incluido f₀).
        n_iteraciones, motivo_corte, tiempo_segundos: como en
            `modelo.ResultadoEntrenamiento`.
    """

    U: np.ndarray
    V: np.ndarray
    b: np.ndarray
    c: np.ndarray
    mu: float
    historial_f: list[float]
    n_iteraciones: int
    motivo_corte: str
    tiempo_segundos: float


def entrenar_als_sesgos(
    R: np.ndarray,
    M: np.ndarray,
    U0: np.ndarray,
    V0: np.ndarray,
    mu: float,
    epsilon: float,
    max_iter: int,
    lambda_: float,
) -> ResultadoSesgos:
    """ALS con sesgos: alterna `paso_usuarios` y `paso_peliculas`.

    Parte de U0, V0 y de b = 0, c = 0, con μ fijo (el promedio de las
    calificaciones de entrenamiento, ver `calcular_mu`). Corta cuando
    |f(t+1) − f(t)| < epsilon, con f = `f_sesgos`, o al llegar a
    `max_iter`.
    """
    inicio = time.perf_counter()
    m, n = R.shape
    U, V = U0, V0
    b, c = np.zeros(m), np.zeros(n)
    f_actual = f_sesgos(R, M, U, V, b, c, mu, lambda_)
    historial_f = [f_actual]

    n_iteraciones = 0
    motivo_corte = "max_iter"
    while n_iteraciones < max_iter:
        # Alternar el paso de usuarios y el de películas.
        U, b = paso_usuarios(R, M, V, c, mu, lambda_)
        V, c = paso_peliculas(R, M, U, b, mu, lambda_)

        f_siguiente = f_sesgos(R, M, U, V, b, c, mu, lambda_)
        n_iteraciones += 1
        historial_f.append(f_siguiente)

        LOGGER.debug("ALS con sesgos iteración %d: f=%.6g", n_iteraciones, f_siguiente)
        if n_iteraciones % 100 == 0:
            LOGGER.info("ALS con sesgos iteración %d: f=%.6g", n_iteraciones, f_siguiente)

        if abs(f_siguiente - f_actual) < epsilon:
            motivo_corte = "tolerancia"
            break
        f_actual = f_siguiente

    if motivo_corte == "max_iter":
        LOGGER.warning("ALS con sesgos: se alcanzó max_iter=%d sin cortar por tolerancia", max_iter)

    return ResultadoSesgos(
        U=U, V=V, b=b, c=c, mu=mu, historial_f=historial_f,
        n_iteraciones=n_iteraciones, motivo_corte=motivo_corte,
        tiempo_segundos=time.perf_counter() - inicio,
    )
