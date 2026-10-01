"""Recomendaciones y extremos por factor latente."""

import numpy as np

from src.modelo import predecir


def recomendar_top_n(
    U: np.ndarray,
    V: np.ndarray,
    M: np.ndarray,
    indice_usuario: int,
    n: int,
    titulos_por_indice: dict[int, str],
) -> list[tuple[str, float]]:
    """Top-n películas no calificadas por el usuario, por mayor r̂ᵢⱼ.

    Entre las películas j con `M[indice_usuario, j] = False`, devuelve las n
    con mayor r̂ᵢⱼ, junto con su título.
    """
    r_hat_usuario = predecir(U, V)[indice_usuario, :]
    candidatos = [j for j in range(V.shape[0]) if not M[indice_usuario, j]]
    candidatos_ordenados = sorted(candidatos, key=lambda j: r_hat_usuario[j], reverse=True)

    return [(titulos_por_indice[j], float(r_hat_usuario[j])) for j in candidatos_ordenados[:n]]


def extremos_por_factor(
    V: np.ndarray, titulos_por_indice: dict[int, str], cantidad: int = 5
) -> list[tuple[int, list[tuple[str, float]], list[tuple[str, float]]]]:
    """Extremos de cada factor latente de V, sin interpretarlos.

    Para cada columna (factor latente) de V, devuelve las `cantidad`
    películas con mayor y con menor valor en esa columna, con título, sin
    etiquetar el factor (la interpretación la hace el grupo en el oral).
    """
    resultado: list[tuple[int, list[tuple[str, float]], list[tuple[str, float]]]] = []
    n_factores = V.shape[1]

    for factor in range(n_factores):
        valores = V[:, factor]
        indices_ordenados = np.argsort(valores)
        indices_menores = indices_ordenados[:cantidad]
        indices_mayores = indices_ordenados[::-1][:cantidad]

        mayores = [(titulos_por_indice[j], float(valores[j])) for j in indices_mayores]
        menores = [(titulos_por_indice[j], float(valores[j])) for j in indices_menores]
        resultado.append((factor, mayores, menores))

    return resultado
