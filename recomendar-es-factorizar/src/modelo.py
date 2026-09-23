"""Módulo compartido entre ALS y descenso de gradiente."""

from dataclasses import dataclass

import numpy as np


def predecir(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Calcula la estimación R̂ = U·Vᵀ (informe sección 4)."""
    return U @ V.T


def sce(R: np.ndarray, M: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
    """Suma de cuadrados del error sobre Ω, sin factor 1/2 (informe 3.5).

    SCE(U,V) = Σ_{(i,j)∈Ω} (rᵢⱼ − uᵢ·vⱼ)². Solo suma sobre las posiciones
    donde M es True; los valores de R fuera de Ω (incluido NaN) no
    intervienen en el resultado.
    """
    error = np.where(M, R - predecir(U, V), 0.0)
    return float(np.sum(error**2))


def inicializar_factores(
    m: int, n: int, k: int, escala: float, generador: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Genera U₀ (m×k) y V₀ (n×k) uniformes en [0, escala) (informe sección 4).

    Función compartida: quien orqueste la comparación (`demo.py`) la llama
    una sola vez y pasa el mismo par (o copias) a `entrenar_als` y
    `entrenar_gd`, para que ambos arranquen de la misma U₀, V₀.
    """
    U = generador.uniform(0.0, escala, size=(m, k))
    V = generador.uniform(0.0, escala, size=(n, k))
    return U, V


@dataclass
class ResultadoEntrenamiento:
    """Resultado de entrenar ALS o descenso de gradiente.

    Atributos:
        U, V: factores finales.
        historial_f: valor de `sce` en cada iteración (incluido f₀).
        n_iteraciones: cantidad de iteraciones realizadas.
        tiempo_segundos: tiempo total de entrenamiento.
        motivo_corte: "tolerancia" o "max_iter".
    """

    U: np.ndarray
    V: np.ndarray
    historial_f: list[float]
    n_iteraciones: int
    tiempo_segundos: float
    motivo_corte: str
