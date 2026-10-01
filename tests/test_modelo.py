"""Predicción, SCE e inicialización compartida."""

import numpy as np

from src.modelo import inicializar_factores, predecir, sce


def test_predecir_calcula_u_por_vt():
    U = np.array([[1.0, 2.0], [3.0, 4.0]])
    V = np.array([[5.0, 6.0], [7.0, 8.0]])

    R_hat = predecir(U, V)

    np.testing.assert_allclose(R_hat, U @ V.T)


def test_sce_ignora_valores_fuera_de_omega():
    R = np.array([[1.0, np.nan], [3.0, 4.0]])
    M = np.array([[True, False], [True, True]])
    U = np.array([[1.0], [2.0]])
    V = np.array([[1.0], [1.0]])

    f_original = sce(R, M, U, V)

    # (0,1) está fuera de Omega: reemplazar el NaN por un número no puede
    # cambiar f, porque sce solo suma sobre las posiciones donde M es True.
    R_modificado = R.copy()
    R_modificado[0, 1] = 999.0
    f_modificado = sce(R_modificado, M, U, V)

    assert f_original == f_modificado
    assert f_original == 5.0  # (1,0): (3-2)^2=1; (1,1): (4-2)^2=4; (0,0): (1-1)^2=0


def test_inicializar_factores_reproducible_con_la_misma_semilla():
    U1, V1 = inicializar_factores(
        m=4, n=5, k=2, escala=2.0, generador=np.random.default_rng(42)
    )
    U2, V2 = inicializar_factores(
        m=4, n=5, k=2, escala=2.0, generador=np.random.default_rng(42)
    )

    assert U1.shape == (4, 2)
    assert V1.shape == (5, 2)
    assert (U1 >= 0).all() and (U1 < 2.0).all()
    assert (V1 >= 0).all() and (V1 < 2.0).all()
    np.testing.assert_array_equal(U1, U2)
    np.testing.assert_array_equal(V1, V2)
