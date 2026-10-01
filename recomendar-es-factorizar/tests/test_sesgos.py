"""Modelo con sesgos por usuario y por película."""

import numpy as np
import pytest

from src.errores import LambdaNegativoError
from src.modelo import f_regularizada
from src.sesgos import (
    calcular_mu,
    f_sesgos,
    predecir_con_sesgos,
    puntajes_ordenamiento_a,
    puntajes_ordenamiento_b,
)


def _caso_chico():
    # 2 usuarios × 3 películas; (0,2) y (1,0) son huecos.
    R = np.array([[4.0, 2.0, np.nan], [np.nan, 5.0, 3.0]])
    M = ~np.isnan(R)
    U = np.array([[1.0, 0.0], [0.5, 1.0]])
    V = np.array([[1.0, 1.0], [0.0, 2.0], [1.0, 0.0]])
    b = np.array([0.5, -0.5])
    c = np.array([0.0, 1.0, -1.0])
    return R, M, U, V, b, c


# --- F con sesgos y predicción ---


def test_calcular_mu_promedia_solo_omega():
    R = np.array([[4.0, 2.0, np.nan], [100.0, 5.0, 3.0]])
    M = np.array([[True, True, False], [False, True, True]])  # el 100 queda fuera de M

    assert calcular_mu(R, M) == pytest.approx((4.0 + 2.0 + 5.0 + 3.0) / 4)


def test_predecir_con_sesgos_suma_mu_b_c_y_u_por_vt():
    _, _, U, V, b, c = _caso_chico()
    # U·Vᵀ = [[1, 0, 1], [1.5, 2, 0.5]]
    esperado = np.array(
        [
            [3.0 + 0.5 + 0.0 + 1.0, 3.0 + 0.5 + 1.0 + 0.0, 3.0 + 0.5 - 1.0 + 1.0],
            [3.0 - 0.5 + 0.0 + 1.5, 3.0 - 0.5 + 1.0 + 2.0, 3.0 - 0.5 - 1.0 + 0.5],
        ]
    )

    np.testing.assert_allclose(predecir_con_sesgos(U, V, b, c, mu=3.0), esperado)


def test_f_sesgos_con_sesgos_y_mu_en_cero_es_f_regularizada():
    R, M, U, V, _, _ = _caso_chico()
    ceros_b, ceros_c = np.zeros(2), np.zeros(3)

    for lambda_ in (0.0, 0.7):
        assert f_sesgos(R, M, U, V, ceros_b, ceros_c, 0.0, lambda_) == f_regularizada(
            R, M, U, V, lambda_
        )  # igualdad exacta


def test_f_sesgos_calculada_a_mano():
    R, M, U, V, b, c = _caso_chico()
    mu, lambda_ = 3.0, 0.5
    # r̂ en Ω: (0,0) = 4.5, (0,1) = 4.5, (1,1) = 5.5, (1,2) = 2.0
    sce_esperada = (4.0 - 4.5) ** 2 + (2.0 - 4.5) ** 2 + (5.0 - 5.5) ** 2 + (3.0 - 2.0) ** 2
    # Σ‖U_i‖² = 1 + 1.25, Σ‖V_j‖² = 2 + 4 + 1, Σ b² = 0.5, Σ c² = 2
    penalizacion = 2.25 + 7.0 + 0.5 + 2.0

    assert f_sesgos(R, M, U, V, b, c, mu, lambda_) == pytest.approx(
        sce_esperada + lambda_ * penalizacion
    )


def test_f_sesgos_ignora_valores_fuera_de_omega():
    R, M, U, V, b, c = _caso_chico()
    R_modificada = R.copy()
    R_modificada[0, 2] = 1000.0  # fuera de Ω

    assert f_sesgos(R_modificada, M, U, V, b, c, 3.0, 0.5) == f_sesgos(R, M, U, V, b, c, 3.0, 0.5)


def test_f_sesgos_con_lambda_negativo_lanza_lambda_negativo_error():
    R, M, U, V, b, c = _caso_chico()

    with pytest.raises(LambdaNegativoError):
        f_sesgos(R, M, U, V, b, c, 3.0, -1.0)


# --- Ordenamientos A y B ---


def test_ordenamiento_a_es_r_hat_completo():
    _, _, U, V, b, c = _caso_chico()

    np.testing.assert_array_equal(
        puntajes_ordenamiento_a(U, V, b, c, mu=3.0), predecir_con_sesgos(U, V, b, c, mu=3.0)
    )


def test_ordenamiento_b_no_depende_de_c():
    _, _, U, V, b, c = _caso_chico()
    otro_c = np.array([5.0, -3.0, 2.0])

    np.testing.assert_array_equal(puntajes_ordenamiento_b(U, V), U @ V.T)
    # B no recibe c; A sí cambia si cambia c
    assert not np.array_equal(
        puntajes_ordenamiento_a(U, V, b, c, 3.0), puntajes_ordenamiento_a(U, V, b, otro_c, 3.0)
    )


def test_ordenamiento_b_da_el_mismo_orden_por_usuario_que_r_hat_sin_c():
    _, _, U, V, b, _ = _caso_chico()
    sin_c = puntajes_ordenamiento_a(U, V, b, np.zeros(3), mu=3.0)  # μ + bᵢ + Uᵢ·Vⱼ

    for i in range(U.shape[0]):
        assert np.array_equal(
            np.argsort(-puntajes_ordenamiento_b(U, V)[i], kind="stable"),
            np.argsort(-sin_c[i], kind="stable"),
        )
