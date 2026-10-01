"""Los defaults de config.py cumplen el criterio de calibración
sobre MovieLens latest-small real (experimentos/bitacora.md, "Recalibración").

Criterio: con los defaults, ALS y GD cortan por tolerancia, GD sin que f
aumente nunca, y las predicciones fuera de Ω quedan en el rango de las
calificaciones (max|r̂| < 7 para los dos métodos a la vez).
"""

from pathlib import Path

import numpy as np
import pytest

from src import config
from src.als import entrenar_als
from src.datos import preparar_datos_movielens
from src.gradiente import entrenar_gd
from src.modelo import inicializar_factores, predecir

RUTA_MOVIELENS = Path(__file__).resolve().parent.parent / "data" / "ml-latest-small"
MAX_R_HAT_FUERA_DE_OMEGA = 7.0


@pytest.mark.movielens
def test_defaults_de_config_cumplen_el_criterio_de_calibracion_sobre_movielens():
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
        R, M, U0.copy(), V0.copy(), epsilon=config.EPSILON_DEFECTO, max_iter=config.MAX_ITER_DEFECTO
    )
    resultado_gd = entrenar_gd(
        R,
        M,
        U0.copy(),
        V0.copy(),
        eta=config.ETA_DEFECTO,
        epsilon=config.EPSILON_DEFECTO,
        max_iter=config.MAX_ITER_DEFECTO,
    )

    assert resultado_als.motivo_corte == "tolerancia"
    assert resultado_gd.motivo_corte == "tolerancia"
    assert np.all(np.diff(resultado_gd.historial_f) <= 0), "f aumentó en alguna iteración de GD"
    for resultado in (resultado_als, resultado_gd):
        assert np.abs(predecir(resultado.U, resultado.V)[~M]).max() < MAX_R_HAT_FUERA_DE_OMEGA
