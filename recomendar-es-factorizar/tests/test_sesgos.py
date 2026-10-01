"""Modelo con sesgos por usuario y por película."""

import numpy as np
import pytest

from src.errores import LambdaNegativoError
from src.modelo import f_regularizada, inicializar_factores
import src.sesgos as sesgos
from src.sesgos import (
    calcular_mu,
    entrenar_als_sesgos,
    f_sesgos,
    paso_peliculas,
    paso_usuarios,
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


# --- Paso aumentado con resolver_factor ---


def _matriz_paso():
    # 3 usuarios × 4 películas, k = 2, factores bien condicionados.
    R = np.array(
        [
            [5.0, 3.0, np.nan, 1.0],
            [np.nan, 4.0, 2.0, 5.0],
            [2.0, np.nan, 4.5, 3.0],
        ]
    )
    M = ~np.isnan(R)
    U = np.array([[1.0, 0.3], [0.2, 1.1], [0.7, 0.6]])
    V = np.array([[1.0, 0.2], [0.3, 1.1], [0.8, 0.5], [0.1, 0.9]])
    b = np.array([0.4, -0.2, 0.1])
    c = np.array([0.3, -0.1, 0.2, -0.4])
    return R, M, U, V, b, c


def _resolver_a_mano(objetivo, M, F, lambda_):
    """Para cada fila: (F_aᵀ F_a + λ I) x = F_aᵀ y, con F_a = [F | 1] en las columnas observadas."""
    k = F.shape[1]
    solucion = np.empty((objetivo.shape[0], k + 1))
    for i in range(objetivo.shape[0]):
        obs = np.where(M[i])[0]
        F_a = np.hstack([F[obs], np.ones((len(obs), 1))])
        y = objetivo[i, obs]
        solucion[i] = np.linalg.solve(F_a.T @ F_a + lambda_ * np.eye(k + 1), F_a.T @ y)
    return solucion


def test_paso_usuarios_coincide_con_el_sistema_de_k_mas_1_incognitas():
    # Incógnitas (U_i, b_i), objetivo r_ij − μ − c_j.
    R, M, _, V, _, c = _matriz_paso()
    mu, lambda_ = 3.2, 0.7

    U, b = paso_usuarios(R, M, V, c, mu, lambda_)

    esperado = _resolver_a_mano(R - mu - c[None, :], M, V, lambda_)
    assert np.allclose(U, esperado[:, :2], rtol=1e-10)
    assert np.allclose(b, esperado[:, 2], rtol=1e-10)


def test_paso_peliculas_coincide_con_el_sistema_de_k_mas_1_incognitas():
    # Incógnitas (V_j, c_j), objetivo r_ij − μ − b_i.
    R, M, U, _, b, _ = _matriz_paso()
    mu, lambda_ = 3.2, 0.7

    V, c = paso_peliculas(R, M, U, b, mu, lambda_)

    esperado = _resolver_a_mano((R - mu - b[:, None]).T, M.T, U, lambda_)
    assert np.allclose(V, esperado[:, :2], rtol=1e-10)
    assert np.allclose(c, esperado[:, 2], rtol=1e-10)


def test_los_dos_pasos_usan_resolver_factor(monkeypatch):
    R, M, U, V, b, c = _matriz_paso()
    llamadas = []
    resolver_real = sesgos.resolver_factor

    def resolver_espia(R_, M_, F, lambda_=0.0):
        llamadas.append(F.copy())
        return resolver_real(R_, M_, F, lambda_)

    monkeypatch.setattr(sesgos, "resolver_factor", resolver_espia)

    paso_usuarios(R, M, V, c, 3.2, 0.7)
    paso_peliculas(R, M, U, b, 3.2, 0.7)

    assert len(llamadas) == 2
    F_usuarios, F_peliculas = llamadas
    np.testing.assert_array_equal(F_usuarios, np.hstack([V, np.ones((4, 1))]))
    np.testing.assert_array_equal(F_peliculas, np.hstack([U, np.ones((3, 1))]))


# --- Entrenamiento ALS con sesgos ---


def _entrenar(R, M, lambda_=0.5, max_iter=100, semilla=3):
    m, n = R.shape
    U0, V0 = inicializar_factores(
        m=m, n=n, k=2, escala=1.0, generador=np.random.default_rng(semilla)
    )
    return entrenar_als_sesgos(
        R, M, U0, V0, mu=calcular_mu(R, M), epsilon=1e-10, max_iter=max_iter, lambda_=lambda_
    )


def test_f_sesgos_no_crece_entre_iteraciones(matriz_ejemplo_informe):
    # Cada paso minimiza f exactamente en sus incógnitas.
    R, M = matriz_ejemplo_informe

    resultado = _entrenar(R, M)

    historial = np.array(resultado.historial_f)
    assert len(historial) > 2
    assert np.all(np.diff(historial) <= 1e-9)


def test_historial_guarda_f_sesgos_del_modelo_final(matriz_ejemplo_informe):
    R, M = matriz_ejemplo_informe

    resultado = _entrenar(R, M, max_iter=7)

    assert resultado.n_iteraciones == 7 and resultado.motivo_corte == "max_iter"
    assert resultado.historial_f[-1] == pytest.approx(
        f_sesgos(R, M, resultado.U, resultado.V, resultado.b, resultado.c, resultado.mu, 0.5)
    )
    assert resultado.mu == calcular_mu(R, M)


def test_entrenar_als_sesgos_es_reproducible(matriz_ejemplo_informe):
    R, M = matriz_ejemplo_informe

    a = _entrenar(R, M)
    b = _entrenar(R, M)

    for x, y in [(a.U, b.U), (a.V, b.V), (a.b, b.b), (a.c, b.c)]:
        assert np.array_equal(x, y)
    assert a.historial_f == b.historial_f


def test_entrenar_als_sesgos_aprende_sesgos_no_nulos():
    # El usuario 0 califica todo con 5 y el resto reparte: su sesgo queda
    # por encima del promedio de los sesgos.
    R = np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [1.0, 3.0, np.nan, 2.0, 4.0],
            [2.0, np.nan, 3.0, 1.0, 3.0],
            [3.0, 2.0, 1.0, np.nan, 2.0],
            [np.nan, 1.0, 2.0, 3.0, 1.0],
        ]
    )
    M = ~np.isnan(R)

    resultado = _entrenar(R, M, lambda_=0.1)

    assert resultado.b[0] > 0
    assert resultado.b[0] > resultado.b[1:].max()
