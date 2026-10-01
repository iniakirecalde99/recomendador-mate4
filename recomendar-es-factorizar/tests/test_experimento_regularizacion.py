"""Métricas, criterios, barrido y
verificación del experimento de regularización."""

import numpy as np
import pytest

from experimentos.regularizacion import evaluar_criterios, medir_fuera_de_rango
from src import config


# --- Métricas ---


def test_fuera_de_rango_cuenta_solo_pares_no_observados():
    # U·Vᵀ = [[10, 1], [1, 10]]: los 10 están fuera de [−0,5; 6].
    U = np.array([[10.0, 1.0], [1.0, 10.0]])
    V = np.eye(2)
    # (0,0) observado (en Ω, sea de entrenamiento o de prueba): no cuenta.
    # (1,1) no observado: cuenta. (0,1) y (1,0) no observados, dentro del rango.
    M = np.array([[True, False], [False, False]])

    fraccion, maximo = medir_fuera_de_rango(U, V, M, escala_min=0.5, escala_max=5.0)

    assert fraccion == pytest.approx(1 / 3)
    assert maximo == pytest.approx(10.0)


def test_fuera_de_rango_usa_los_limites_menos_medio_y_seis():
    # Una película, cuatro usuarios no observados con r̂ = −0,5; 6; −0,51; 6,01.
    U = np.array([[-0.5], [6.0], [-0.51], [6.01]])
    V = np.array([[1.0]])
    M = np.zeros((4, 1), dtype=bool)

    fraccion, maximo = medir_fuera_de_rango(U, V, M, escala_min=0.5, escala_max=5.0)

    assert fraccion == pytest.approx(2 / 4)  # −0,5 y 6 están dentro; −0,51 y 6,01, fuera
    assert maximo == pytest.approx(6.01)


def test_config_tiene_la_escala_y_los_umbrales_de_la_seccion_9():
    assert config.ESCALA_MIN == 0.5
    assert config.ESCALA_MAX == 5.0
    assert config.MIN_PELICULAS_EXITO == 130
    assert config.MAX_FRECUENCIA_EXITO == 0.25
    assert config.MAX_FRACCION_FUERA_DE_RANGO == 0.01
    assert config.TOLERANCIA_EMPATE == 0.01


# --- Criterios de éxito ---


def _caso_exitoso(**cambios):
    valores = dict(
        k=5,
        sce_prueba_elegido=900.0,
        sce_prueba_k2_lambda0=1000.0,
        fraccion_fuera_de_rango=0.005,
        peliculas_en_algun_top=150,
        frecuencia_mas_frecuente=0.20,
    )
    valores.update(cambios)
    return evaluar_criterios(**valores)


def test_evaluar_criterios_exitoso_si_cumple_todo():
    veredicto = _caso_exitoso()

    assert veredicto.exitoso
    assert veredicto.criterios == {1: True, 2: True, 3: True, 4: True}


def test_evaluar_criterios_falla_si_k_es_2():
    veredicto = _caso_exitoso(k=2)

    assert not veredicto.exitoso
    assert veredicto.criterios[1] is False


def test_evaluar_criterios_falla_si_la_sce_de_prueba_no_mejora_a_k2_lambda0():
    assert _caso_exitoso(sce_prueba_elegido=1000.0).criterios[2] is False  # igual no alcanza
    assert not _caso_exitoso(sce_prueba_elegido=1001.0).exitoso


def test_evaluar_criterios_falla_con_mas_de_1_por_ciento_fuera_de_rango():
    assert _caso_exitoso(fraccion_fuera_de_rango=0.01).criterios[3] is True  # "a lo sumo"
    assert _caso_exitoso(fraccion_fuera_de_rango=0.0101).criterios[3] is False


def test_evaluar_criterios_falla_con_menos_de_130_peliculas_o_mas_de_25_por_ciento():
    assert _caso_exitoso(peliculas_en_algun_top=130).criterios[4] is True  # "al menos"
    assert _caso_exitoso(peliculas_en_algun_top=129).criterios[4] is False
    assert _caso_exitoso(frecuencia_mas_frecuente=0.25).criterios[4] is True  # "a lo sumo"
    assert _caso_exitoso(frecuencia_mas_frecuente=0.2501).criterios[4] is False
