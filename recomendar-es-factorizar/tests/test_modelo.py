"""Predicción, SCE e inicialización compartida."""

import numpy as np
import pytest

from src.errores import LambdaNegativoError
from src.modelo import f_regularizada, inicializar_factores, predecir, sce, validar_lambda


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


# --- F regularizada y validar_lambda ---


def _caso_2x2():
    # U·Vᵀ = [[1, 3], [0, 1]]; Ω = {(0,0), (1,0), (1,1)}; (0,1) es hueco.
    R = np.array([[3.0, np.nan], [1.0, 2.0]])
    M = ~np.isnan(R)
    U = np.array([[1.0, 2.0], [0.0, 1.0]])
    V = np.array([[1.0, 0.0], [1.0, 1.0]])
    return R, M, U, V


def test_f_regularizada_con_lambda_cero_es_la_sce():
    R, M, U, V = _caso_2x2()

    assert f_regularizada(R, M, U, V, lambda_=0.0) == sce(R, M, U, V)  # igualdad exacta


def test_f_regularizada_suma_lambda_por_normas_al_cuadrado():
    R, M, U, V = _caso_2x2()
    # SCE sobre Ω: (3-1)² + (1-0)² + (2-1)² = 6
    # Σ‖U_i‖² = 1+4+0+1 = 6, Σ‖V_j‖² = 1+0+1+1 = 3 → λ·9
    lambda_ = 0.5

    assert f_regularizada(R, M, U, V, lambda_=lambda_) == pytest.approx(6.0 + 0.5 * 9.0)


def test_f_regularizada_ignora_valores_fuera_de_omega():
    R, M, U, V = _caso_2x2()
    R_modificada = R.copy()
    R_modificada[0, 1] = 1000.0  # fuera de Ω: no tiene que cambiar f

    assert f_regularizada(R_modificada, M, U, V, lambda_=0.5) == f_regularizada(
        R, M, U, V, lambda_=0.5
    )


def test_f_regularizada_con_lambda_negativo_lanza_lambda_negativo_error():
    R, M, U, V = _caso_2x2()

    with pytest.raises(LambdaNegativoError, match=r"-1"):
        f_regularizada(R, M, U, V, lambda_=-1.0)


def test_validar_lambda_acepta_cero_y_positivos_y_rechaza_negativos():
    validar_lambda(0.0)
    validar_lambda(2.5)

    with pytest.raises(LambdaNegativoError) as exc_info:
        validar_lambda(-0.1)

    assert exc_info.value.lambda_ == -0.1
