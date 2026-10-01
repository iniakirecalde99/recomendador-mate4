"""Experimento de regularización: barrido de (k, λ) y veredicto.

Vive fuera de `src/`: usa `entrenar_als`, `entrenar_gd`, `resolver_factor`,
`f_regularizada` y `particionar` de `src/` (que con λ = 0 son main) y agrega
lo propio del experimento: métricas, criterios de éxito, barrido y
selección y la verificación con descenso de gradiente.

Uso: `python -m experimentos.regularizacion --help`
"""

import logging
from dataclasses import dataclass

import numpy as np

from src import config
from src.modelo import predecir

LOGGER = logging.getLogger(__name__)


# --- Métricas ---


def medir_fuera_de_rango(
    U: np.ndarray, V: np.ndarray, M: np.ndarray, escala_min: float, escala_max: float
) -> tuple[float, float]:
    """Fracción de estimaciones fuera de rango y máximo de |r̂|.

    Mira solo los pares no observados (M False, con M la máscara de todo Ω:
    ni entrenamiento ni prueba). Una estimación está fuera de rango si cae
    fuera de [escala_min − 1, escala_max + 1]; los bordes cuentan como
    dentro. Devuelve (fracción entre 0 y 1, max |r̂| sobre esos pares).
    """
    r_hat_no_observados = predecir(U, V)[~M]
    fuera = (r_hat_no_observados < escala_min - 1) | (r_hat_no_observados > escala_max + 1)
    return float(fuera.mean()), float(np.abs(r_hat_no_observados).max())


# --- Criterios de éxito ---


@dataclass
class Veredicto:
    """Resultado de evaluar los cuatro criterios.

    Atributos:
        criterios: número de criterio (1 a 4) → si se cumple.
    """

    criterios: dict[int, bool]

    @property
    def exitoso(self) -> bool:
        """El experimento es exitoso si se cumplen los cuatro criterios."""
        return all(self.criterios.values())


def evaluar_criterios(
    k: int,
    sce_prueba_elegido: float,
    sce_prueba_k2_lambda0: float,
    fraccion_fuera_de_rango: float,
    peliculas_en_algun_top: int,
    frecuencia_mas_frecuente: float,
    min_peliculas: int = config.MIN_PELICULAS_EXITO,
    max_frecuencia: float = config.MAX_FRECUENCIA_EXITO,
    max_fraccion_fuera: float = config.MAX_FRACCION_FUERA_DE_RANGO,
) -> Veredicto:
    """Evalúa el par elegido contra los criterios de éxito.

    El criterio 2 usa SCE de prueba (de la partición); los criterios 3 y 4
    usan el modelo reentrenado sobre todo Ω.
    1. k > 2.
    2. SCE de prueba del par elegido < la de (k = 2, λ = 0).
    3. Fracción fuera de rango <= max_fraccion_fuera.
    4. Películas en algún top-10 >= min_peliculas y la más frecuente en a lo
       sumo max_frecuencia de los usuarios.
    """
    return Veredicto(
        criterios={
            1: k > 2,
            2: sce_prueba_elegido < sce_prueba_k2_lambda0,
            3: fraccion_fuera_de_rango <= max_fraccion_fuera,
            4: peliculas_en_algun_top >= min_peliculas
            and frecuencia_mas_frecuente <= max_frecuencia,
        }
    )
