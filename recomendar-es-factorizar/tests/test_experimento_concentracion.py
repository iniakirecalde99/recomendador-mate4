"""Concentración con λ.

La concentración se mide con el modelo entrenado con ALS sobre todo Ω, nunca
con el de la partición: `simular` recibe la M completa.
"""

from pathlib import Path

import numpy as np
import pytest

import experimentos.concentracion as concentracion
from experimentos.concentracion import notas_reales, simular
from src import config
from src.datos import preparar_datos_movielens

RUTA_MOVIELENS = Path(__file__).resolve().parent.parent / "data" / "ml-latest-small"


def _simular(R, M, titulos, notas, lambda_, k=2, n_usuarios=3000):
    return simular(
        R, M, titulos, centrar=False, notas=notas,
        k=k, epsilon=config.EPSILON_DEFECTO, max_iter=config.MAX_ITER_DEFECTO,
        semilla=config.SEMILLA_INICIALIZACION, n_usuarios=n_usuarios,
        min_calificadas=5, max_calificadas=10, n_a_calificar=config.N_A_CALIFICAR_DEFECTO,
        top_n=config.TOP_N_DEFECTO, semilla_simulacion=0, umbral_frecuente=0.2,
        lambda_=lambda_,
    )


@pytest.mark.movielens
def test_simular_con_lambda_cero_reproduce_la_bitacora():
    datos = preparar_datos_movielens(
        RUTA_MOVIELENS / "ratings.csv", RUTA_MOVIELENS / "movies.csv",
        config.UMBRAL_DEFECTO, config.K_DEFECTO,
    )
    R, M = datos.calificaciones.R, datos.calificaciones.M

    resultado = _simular(
        R, M, datos.titulos_por_indice, notas_reales(RUTA_MOVIELENS / "ratings.csv"), lambda_=0.0
    )

    # Bitácora, experimento de concentración: notas reales, sin centrado.
    assert resultado.distintas_en_top_n == 65
    assert round(100 * resultado.frecuencia_mas_frecuente, 1) == 44.5


def test_simular_pasa_lambda_al_entrenamiento_y_a_resolver_factor(monkeypatch):
    generador = np.random.default_rng(2)
    m, n = 15, 20
    M = generador.uniform(size=(m, n)) < 0.7
    R = np.where(M, generador.integers(1, 6, size=(m, n)).astype(float), np.nan)
    titulos = {j: f"película {j}" for j in range(n)}
    lambdas_entrenamiento, lambdas_usuario = [], []
    entrenar_real, resolver_real = concentracion.entrenar_als, concentracion.resolver_factor

    def entrenar_espia(*args, lambda_=0.0, **kwargs):
        lambdas_entrenamiento.append(lambda_)
        return entrenar_real(*args, lambda_=lambda_, **kwargs)

    def resolver_espia(*args, lambda_=0.0):
        lambdas_usuario.append(lambda_)
        return resolver_real(*args, lambda_=lambda_)

    monkeypatch.setattr(concentracion, "entrenar_als", entrenar_espia)
    monkeypatch.setattr(concentracion, "resolver_factor", resolver_espia)

    _simular(R, M, titulos, notas=None, lambda_=5.0, n_usuarios=20)

    assert lambdas_entrenamiento == [5.0]
    assert lambdas_usuario == [5.0] * 20


def test_parser_de_concentracion_tiene_lambda_con_default_cero():
    args = concentracion.construir_parser().parse_args([])
    assert args.lambda_ == 0.0
    assert concentracion.construir_parser().parse_args(["--lambda", "5"]).lambda_ == 5.0
