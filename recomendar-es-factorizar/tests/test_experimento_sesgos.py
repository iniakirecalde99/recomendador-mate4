"""Precisión@10, barrido, reentrenamiento y
criterio del experimento de sesgos."""

import numpy as np
import pytest

from experimentos.sesgos import precision_en_n
from src import config


# --- Precisión@N ---


def _caso_precision():
    # 3 usuarios × 5 películas. Puntajes ordenan 0 > 1 > 2 > 3 > 4 para todos.
    puntajes = np.tile(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), (3, 1))
    R = np.array(
        [
            [3.0, 4.5, 2.0, np.nan, 4.0],
            [5.0, 1.0, np.nan, 4.0, np.nan],
            [2.0, 3.0, 3.5, np.nan, 1.0],
        ]
    )
    M_ent = np.array(
        [
            [True, False, False, False, False],  # usuario 0: la 0 está en entrenamiento
            [False, True, False, False, False],
            [True, True, False, False, False],
        ]
    )
    M_prueba = ~np.isnan(R) & ~M_ent
    return puntajes, M_ent, R, M_prueba


def test_precision_en_n_calculada_a_mano():
    # top-2:
    # usuario 0: sin la 0 (entrenamiento), top-2 = {1, 2}; relevantes en
    #   prueba: 1 (4,5) y 4 (4,0) → acierta 1 → 1/2.
    # usuario 1: sin la 1, top-2 = {0, 2}; relevantes: 0 (5,0) y 3 (4,0) → 1/2.
    # usuario 2: en prueba tiene 2 (3,5) y 4 (1,0): ninguna relevante → afuera.
    puntajes, M_ent, R, M_prueba = _caso_precision()

    precision = precision_en_n(puntajes, M_ent, R, M_prueba, top_n=2, umbral_relevante=4.0)

    assert precision == pytest.approx((0.5 + 0.5) / 2)


def test_precision_excluye_del_top_lo_que_esta_en_entrenamiento():
    puntajes, M_ent, R, M_prueba = _caso_precision()
    # Si la película 0 del usuario 1 pasa a entrenamiento, deja de contar
    # como acierto y sale de su top.
    M_ent = M_ent.copy()
    M_ent[1, 0] = True
    M_prueba = M_prueba.copy()
    M_prueba[1, 0] = False

    precision = precision_en_n(puntajes, M_ent, R, M_prueba, top_n=2, umbral_relevante=4.0)

    # usuario 0: 1/2; usuario 1: top-2 = {2, 3}, relevante la 3 → 1/2.
    assert precision == pytest.approx(0.5)


def test_relevante_es_mayor_o_igual_que_el_umbral():
    puntajes = np.array([[2.0, 1.0]])
    M_ent = np.zeros((1, 2), dtype=bool)
    M_prueba = np.ones((1, 2), dtype=bool)

    con_cuatro = precision_en_n(
        puntajes, M_ent, np.array([[4.0, 1.0]]), M_prueba, top_n=1, umbral_relevante=4.0
    )
    con_tres_y_medio = precision_en_n(
        puntajes, M_ent, np.array([[3.5, 4.5]]), M_prueba, top_n=1, umbral_relevante=4.0
    )

    assert con_cuatro == 1.0  # 4,0 cuenta como relevante
    assert con_tres_y_medio == 0.0  # 3,5 no; el usuario cuenta por el 4,5 que no está en su top-1


def test_config_tiene_grilla_y_umbrales_del_experimento():
    assert config.UMBRAL_RELEVANTE == 4.0
    assert config.GRILLA_K_SESGOS == (2, 5, 10, 20)
    assert config.GRILLA_LAMBDA_SESGOS == (1.0, 5.0, 10.0, 20.0)
    assert config.MAX_FRECUENCIA_SESGOS == 0.25
    assert config.MIN_PELICULAS_DISTINTAS_SESGOS == 127
    assert config.MAX_FRACCION_FUERA_DE_RANGO_SESGOS == 0.01
    # las de la regularización no se tocan
    assert config.GRILLA_K == (2, 3, 5, 10)
    assert config.MIN_PELICULAS_EXITO == 130
