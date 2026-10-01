"""Garantía de main con λ = 0.

Si este test no pasa, el cambio en src/ rompió algo y la rama no sigue.
"""

from pathlib import Path

import numpy as np
import pytest

from src import config
from src.als import entrenar_als
from src.datos import preparar_datos_movielens
from src.gradiente import entrenar_gd
from src.modelo import inicializar_factores

RUTA_MOVIELENS = Path(__file__).resolve().parent.parent / "data" / "ml-latest-small"


@pytest.mark.movielens
def test_lambda_cero_sin_particion_reproduce_main():
    datos = preparar_datos_movielens(
        RUTA_MOVIELENS / "ratings.csv",
        RUTA_MOVIELENS / "movies.csv",
        config.UMBRAL_DEFECTO,
        config.K_DEFECTO,
    )
    R = datos.calificaciones.R
    M = datos.calificaciones.M
    m, n = R.shape
    U0, V0 = inicializar_factores(
        m=m,
        n=n,
        k=config.K_DEFECTO,
        escala=config.ESCALA_INICIALIZACION_DEFECTO,
        generador=np.random.default_rng(config.SEMILLA_DEFECTO),
    )

    resultado_als = entrenar_als(
        R, M, U0.copy(), V0.copy(),
        epsilon=config.EPSILON_DEFECTO, max_iter=config.MAX_ITER_DEFECTO, lambda_=0.0,
    )
    resultado_gd = entrenar_gd(
        R, M, U0.copy(), V0.copy(),
        eta=config.ETA_DEFECTO, epsilon=config.EPSILON_DEFECTO,
        max_iter=config.MAX_ITER_DEFECTO, lambda_=0.0,
    )

    # Números de main (bitácora): exactos a 4 decimales y mismas iteraciones.
    assert resultado_als.n_iteraciones == 10
    assert round(resultado_als.historial_f[-1], 4) == 22937.3886
    assert resultado_gd.n_iteraciones == 462
    assert round(resultado_gd.historial_f[-1], 4) == 23045.8796
