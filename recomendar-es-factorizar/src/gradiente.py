"""Descenso de gradiente: versión genérica y gradiente de la SCE (informe 3.4 y 6)."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.modelo import predecir

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
