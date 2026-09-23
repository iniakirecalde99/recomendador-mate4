"""Descenso de gradiente: versión genérica y descenso completo sobre la factorización (informe 3.4 y 6)."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.errores import DivergenciaError
from src.modelo import ResultadoEntrenamiento, predecir, sce

LOGGER = logging.getLogger(__name__)


@dataclass
class ResultadoDescensoGenerico:
    """Resultado de `descenso_gradiente`.

    Atributos:
        x: punto final.
        historial_f: valor de f en cada iteración (incluido f₀).
        n_iteraciones: cantidad de iteraciones realizadas.
        motivo_corte: "tolerancia" o "max_iter".
    """

    x: np.ndarray
    historial_f: list[float]
    n_iteraciones: int
    motivo_corte: str


def descenso_gradiente(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    eta: float,
    epsilon: float,
    max_iter: int,
) -> ResultadoDescensoGenerico:
    """Descenso de gradiente genérico para funciones de ℝⁿ (informe 3.4).

    En cada iteración actualiza x ← x − eta · grad_f(x) hasta que
    |f(t+1) − f(t)| < epsilon o se alcanza `max_iter`.
    """
    x = np.array(x0, dtype=float)
    f_actual = f(x)
    historial_f = [f_actual]

    n_iteraciones = 0
    motivo_corte = "max_iter"
    while n_iteraciones < max_iter:
        x = x - eta * grad_f(x)
        f_siguiente = f(x)
        n_iteraciones += 1
        historial_f.append(f_siguiente)

        if f_siguiente > f_actual:
            LOGGER.warning(
                "descenso_gradiente: f aumentó en la iteración %d (%.6g -> %.6g); "
                "eta=%.4g podría ser demasiado grande",
                n_iteraciones, f_actual, f_siguiente, eta,
            )

        if abs(f_siguiente - f_actual) < epsilon:
            motivo_corte = "tolerancia"
            f_actual = f_siguiente
            break

        f_actual = f_siguiente

    if motivo_corte == "max_iter":
        LOGGER.warning(
            "descenso_gradiente: se alcanzó max_iter=%d sin cortar por tolerancia", max_iter
        )

    return ResultadoDescensoGenerico(
        x=x,
        historial_f=historial_f,
        n_iteraciones=n_iteraciones,
        motivo_corte=motivo_corte,
    )


def gradiente_sce(
    R: np.ndarray, M: np.ndarray, U: np.ndarray, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula ∇_U f y ∇_V f de la SCE, sin factor 1/2 (informe 3.4/6).

    Con E = M ⊙ (R − U·Vᵀ) (error solo sobre Ω, cero fuera):
    ∇_U f = −2·E·V, ∇_V f = −2·Eᵀ·U, ambos evaluados en el mismo (U, V)
    recibido.
    """
    # Informe 3.4/6: gradientes de la SCE respecto de U y de V.
    E = np.where(M, R - predecir(U, V), 0.0)
    grad_U = -2 * E @ V
    grad_V = -2 * E.T @ U
    return grad_U, grad_V


def entrenar_gd(
    R: np.ndarray,
    M: np.ndarray,
    U0: np.ndarray,
    V0: np.ndarray,
    eta: float,
    epsilon: float,
    max_iter: int,
) -> ResultadoEntrenamiento:
    """Descenso de gradiente completo sobre la factorización (informe 3.4 y 6).

    En cada iteración calcula `gradiente_sce` con (U, V) de la iteración t y
    actualiza ambos simultáneamente: U ← U − eta·∇_U f, V ← V − eta·∇_V f
    (nunca actualizar U y recién ahí calcular el
    gradiente de V con la U nueva, eso lo convertiría en un método
    alternado). Usa `modelo.sce` para el criterio de corte.
    """
    inicio = time.perf_counter()

    U, V = U0, V0
    f_actual = sce(R, M, U, V)
    historial_f = [f_actual]

    n_iteraciones = 0
    motivo_corte = "max_iter"
    while n_iteraciones < max_iter:
        grad_U, grad_V = gradiente_sce(R, M, U, V)
        # Informe 6: x(t+1) = x(t) - eta * grad f(x(t)), U y V con el
        # gradiente de la iteración t
        U = U - eta * grad_U
        V = V - eta * grad_V

        f_siguiente = sce(R, M, U, V)
        n_iteraciones += 1

        if not np.isfinite(f_siguiente):
            raise DivergenciaError(iteracion=n_iteraciones, eta=eta)

        historial_f.append(f_siguiente)

        LOGGER.debug("GD iteración %d: f=%.6g", n_iteraciones, f_siguiente)
        if n_iteraciones % 100 == 0:
            LOGGER.info("GD iteración %d: f=%.6g", n_iteraciones, f_siguiente)

        if f_siguiente > f_actual:
            LOGGER.warning(
                "entrenar_gd: f aumentó en la iteración %d (%.6g -> %.6g); "
                "eta=%.4g podría ser demasiado grande",
                n_iteraciones, f_actual, f_siguiente, eta,
            )

        if abs(f_siguiente - f_actual) < epsilon:
            motivo_corte = "tolerancia"
            f_actual = f_siguiente
            break

        f_actual = f_siguiente

    if motivo_corte == "max_iter":
        LOGGER.warning(
            "entrenar_gd: se alcanzó max_iter=%d sin cortar por tolerancia", max_iter
        )

    LOGGER.info(
        "GD terminó: %d iteraciones (motivo=%s), f final=%.6g",
        n_iteraciones, motivo_corte, historial_f[-1],
    )

    return ResultadoEntrenamiento(
        U=U,
        V=V,
        historial_f=historial_f,
        n_iteraciones=n_iteraciones,
        tiempo_segundos=time.perf_counter() - inicio,
        motivo_corte=motivo_corte,
    )
