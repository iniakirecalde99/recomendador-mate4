"""Experimento de la sección 8: división por usuario y RMSE."""

import numpy as np

from experimentos.seccion8 import dividir_por_usuario, rmse
from src.modelo import sce


def test_dividir_por_usuario_es_disjunta_y_deja_al_menos_k_en_entrenamiento():
    k = 2
    umbral = 10  # como después del filtro: toda fila tiene al menos `umbral` >= k observaciones
    generador = np.random.default_rng(0)
    M = generador.random((8, 30)) < 0.5
    M[:, :umbral] = True  # garantiza el mínimo por fila

    M_ent, M_pru = dividir_por_usuario(M, 0.2, k, np.random.default_rng(42))

    assert np.array_equal(M_ent | M_pru, M)
    assert not np.any(M_ent & M_pru)
    for i in range(M.shape[0]):
        assert M_pru[i].sum() == round(0.2 * M[i].sum())
        assert M_ent[i].sum() >= k

    # Misma semilla, misma división.
    M_ent_2, M_pru_2 = dividir_por_usuario(M, 0.2, k, np.random.default_rng(42))
    assert np.array_equal(M_ent, M_ent_2)
    assert np.array_equal(M_pru, M_pru_2)


def test_dividir_por_usuario_no_deja_a_nadie_con_menos_de_k_en_entrenamiento():
    # Con 3 calificaciones y la mitad a prueba, round(1,5) = 2 dejaría 1 < k en
    # entrenamiento: la división se limita a mandar a prueba solo 3 − k = 1.
    k = 2
    M = np.zeros((2, 6), dtype=bool)
    M[0, :3] = True
    M[1, :] = True

    M_ent, M_pru = dividir_por_usuario(M, 0.5, k, np.random.default_rng(42))

    assert M_ent[0].sum() == k
    assert M_pru[0].sum() == 1
    assert M_pru[1].sum() == 3


def test_rmse_coincide_con_la_raiz_de_sce_sobre_la_mascara():
    # A mano: errores 1, 0 y 2 sobre Ω; el NaN y el 9 de fuera de Ω no cuentan.
    R = np.array([[1.0, np.nan], [3.0, 4.0]])
    R_hat = np.array([[2.0, 9.0], [3.0, 2.0]])
    M = ~np.isnan(R)
    assert np.isclose(rmse(R, R_hat, M), np.sqrt(5 / 3))

    # RMSE = √(SCE / |Ω|), con la SCE de src.modelo.
    generador = np.random.default_rng(1)
    U = generador.random((5, 2))
    V = generador.random((7, 2))
    M = generador.random((5, 7)) < 0.6
    R = np.where(M, generador.uniform(0.5, 5.0, size=(5, 7)), np.nan)
    assert np.isclose(rmse(R, U @ V.T, M), np.sqrt(sce(R, M, U, V) / M.sum()))
