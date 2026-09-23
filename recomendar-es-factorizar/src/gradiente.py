"""Descenso de gradiente genérico (informe 3.4)."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

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
