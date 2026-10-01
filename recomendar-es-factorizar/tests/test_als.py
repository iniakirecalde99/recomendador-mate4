"""resolver_factor y entrenar_als (informe 3.5)."""

import logging

import numpy as np
import pytest

from src.als import entrenar_als, resolver_factor
from src.errores import SistemaSingularError
from src.modelo import inicializar_factores


def test_resolver_factor_transpuesto_coincide_con_calculo_manual_de_v():
    # 3 usuarios x 3 películas, k=2.
    R = np.array(
        [
            [5.0, 4.0, np.nan],
            [3.0, np.nan, 2.0],
            [1.0, 5.0, 4.0],
        ]
    )
    M = np.array(
        [
            [True, True, False],
            [True, False, True],
            [True, True, True],
        ]
    )
    U = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    V = resolver_factor(R.T, M.T, U)

    # Paso de V fila por fila a mano, sin transponer: para cada película j,
    # arma las ecuaciones normales con los usuarios que la calificaron.
    n_peliculas = R.shape[1]
    k = U.shape[1]
    V_manual = np.empty((n_peliculas, k))
    for j in range(n_peliculas):
        usuarios_obs = np.where(M[:, j])[0]
        U_obs = U[usuarios_obs, :]
        r_obs = R[usuarios_obs, j]
        V_manual[j, :] = np.linalg.solve(U_obs.T @ U_obs, U_obs.T @ r_obs)

    np.testing.assert_allclose(V, V_manual)


def test_resolver_factor_lanza_sistema_singular_si_hay_menos_datos_que_incognitas():
    # Fila 0: una sola columna observada para k=2 incógnitas -> sistema singular.
    R = np.array([[5.0, np.nan], [3.0, 4.0]])
    M = np.array([[True, False], [True, True]])
    F = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(SistemaSingularError) as exc_info:
        resolver_factor(R, M, F)

    assert exc_info.value.fila == 0
    assert exc_info.value.n_observados == 1


def test_historial_de_sce_de_als_no_crece(matriz_ejemplo_informe, generador_fijo):
    R, M = matriz_ejemplo_informe
    U0, V0 = inicializar_factores(m=4, n=5, k=2, escala=1.0, generador=generador_fijo)

    resultado = entrenar_als(R, M, U0, V0, epsilon=1e-8, max_iter=100)

    historial = np.array(resultado.historial_f)
    assert np.all(np.diff(historial) <= 1e-9)  # no creciente, con margen numérico


def test_als_ejemplo_4x5_no_lanza_singular_y_sce_decrece(matriz_ejemplo_informe, generador_fijo):
    R, M = matriz_ejemplo_informe
    U0, V0 = inicializar_factores(m=4, n=5, k=2, escala=1.0, generador=generador_fijo)

    resultado = entrenar_als(R, M, U0, V0, epsilon=1e-8, max_iter=100)

    assert resultado.historial_f[-1] < resultado.historial_f[0]


@pytest.mark.skip(
    reason="V0 del ejemplo de la sección 5 del informe todavía no está fijado"
)
def test_als_ejemplo_4x5_primera_iteracion_coincide_con_informe():
    """Compara la primera iteración de ALS contra la tabla del informe (sección 5).

    Pendiente hasta fijar la V0 exacta que usa ese ejemplo.
    """


def test_entrenar_als_loguea_debug_por_iteracion_e_info_cada_100_y_resumen(
    matriz_ejemplo_informe, generador_fijo, caplog
):
    R, M = matriz_ejemplo_informe
    U0, V0 = inicializar_factores(m=4, n=5, k=2, escala=1.0, generador=generador_fijo)

    # epsilon=0.0 nunca corta por tolerancia (abs(diff) < 0.0 nunca es True):
    # fuerza exactamente max_iter iteraciones, para poder contar los logs.
    with caplog.at_level(logging.DEBUG, logger="src.als"):
        resultado = entrenar_als(R, M, U0, V0, epsilon=0.0, max_iter=250)

    assert resultado.n_iteraciones == 250

    mensajes_debug = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG and "ALS iteración" in r.getMessage()
    ]
    mensajes_info_iteracion = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "ALS iteración" in r.getMessage()
    ]
    mensajes_resumen = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "terminó" in r.getMessage()
    ]

    assert len(mensajes_debug) == 250
    assert len(mensajes_info_iteracion) == 2  # iteraciones 100 y 200 (250 no es múltiplo)
    assert len(mensajes_resumen) == 1


# --- λ en resolver_factor ---


def _ecuaciones_normales_regularizadas_se_cumplen(R, M, F, factor, lambda_):
    """True si cada fila i de `factor` cumple (F_iᵀ F_i + λ I) u_i = F_iᵀ r_i."""
    k = F.shape[1]
    for i in range(R.shape[0]):
        columnas_obs = np.where(M[i, :])[0]
        F_obs = F[columnas_obs, :]
        r_obs = R[i, columnas_obs]
        izquierda = (F_obs.T @ F_obs + lambda_ * np.eye(k)) @ factor[i, :]
        derecha = F_obs.T @ r_obs
        if not np.allclose(izquierda, derecha, rtol=1e-10):
            return False
    return True


def test_resolver_factor_con_lambda_satisface_las_ecuaciones_normales_regularizadas():
    # 3 usuarios x 4 películas, k=2, F bien condicionada.
    R = np.array(
        [
            [5.0, 3.0, np.nan, 1.0],
            [np.nan, 4.0, 2.0, 5.0],
            [2.0, np.nan, 4.5, 3.0],
        ]
    )
    M = ~np.isnan(R)
    F = np.array(
        [
            [1.0, 0.2],
            [0.3, 1.1],
            [0.8, 0.5],
            [0.1, 0.9],
        ]
    )
    lambda_ = 0.7

    U = resolver_factor(R, M, F, lambda_=lambda_)

    assert _ecuaciones_normales_regularizadas_se_cumplen(R, M, F, U, lambda_)
    # con λ > 0 la solución no es la de mínimos cuadrados sin regularizar
    assert not np.allclose(U, resolver_factor(R, M, F))


def test_resolver_factor_con_lambda_positivo_no_es_singular_con_menos_de_k_calificaciones():
    # La fila 0 tiene 1 calificación para k=2 incógnitas; sin λ es
    # singular (test anterior), con λ > 0 el sistema tiene solución única.
    R = np.array([[5.0, np.nan], [3.0, 4.0]])
    M = np.array([[True, False], [True, True]])
    F = np.array([[1.0, 2.0], [3.0, 4.0]])
    lambda_ = 1.0

    U = resolver_factor(R, M, F, lambda_=lambda_)

    assert np.all(np.isfinite(U))
    assert _ecuaciones_normales_regularizadas_se_cumplen(R, M, F, U, lambda_)


def test_resolver_factor_con_lambda_cero_es_identico_al_actual():
    R = np.array(
        [
            [5.0, 4.0, np.nan],
            [3.0, np.nan, 2.0],
            [1.0, 5.0, 4.0],
        ]
    )
    M = ~np.isnan(R)
    F = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # igualdad exacta, no aproximada: con λ = 0 tiene que ser el mismo cálculo
    assert np.array_equal(resolver_factor(R, M, F, lambda_=0.0), resolver_factor(R, M, F))
    assert np.array_equal(
        resolver_factor(R.T, M.T, F, lambda_=0.0), resolver_factor(R.T, M.T, F)
    )


def test_resolver_factor_con_lambda_negativo_lanza_value_error_con_el_valor():
    R = np.array([[5.0, 4.0], [3.0, 2.0]])
    M = ~np.isnan(R)
    F = np.array([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match=r"-1"):
        resolver_factor(R, M, F, lambda_=-1.0)
