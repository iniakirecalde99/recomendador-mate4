"""Experimento de sesgos: barrido de (k, λ), precisión@10 y concentración.

Vive fuera de `src/`: usa el modelo con sesgos de `src/sesgos.py` y, como
línea de base, `entrenar_als` de main. Mide SCE de prueba, fuera de rango,
precisión@10 y concentración con los dos ordenamientos, y
evalúa el criterio. Esta rama no usa descenso de gradiente.

Uso: `python -m experimentos.sesgos --help`
"""

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


# --- Precisión@N ---


def precision_en_n(
    puntajes: np.ndarray,
    M_ent: np.ndarray,
    R: np.ndarray,
    M_prueba: np.ndarray,
    top_n: int,
    umbral_relevante: float,
) -> float:
    """Precisión@N promedio sobre los usuarios con algo relevante en prueba.

    Para cada usuario con al menos una calificación >= `umbral_relevante` en
    Ω_prueba, toma el top-N por `puntajes` entre las películas que no tiene
    en Ω_ent (empates por índice) y calcula aciertos / N, donde un acierto es
    una película del top-N calificada >= `umbral_relevante` en Ω_prueba.
    Devuelve el promedio sobre esos usuarios.
    """
    relevantes = M_prueba & (np.where(M_prueba, R, -np.inf) >= umbral_relevante)
    usuarios = np.where(relevantes.any(axis=1))[0]

    puntajes = puntajes.copy()
    puntajes[M_ent] = -np.inf  # lo de entrenamiento no se recomienda
    tops = np.argsort(-puntajes, axis=1, kind="stable")[:, :top_n]

    aciertos = np.take_along_axis(relevantes, tops, axis=1)[usuarios].sum(axis=1)
    return float(np.mean(aciertos / top_n))
