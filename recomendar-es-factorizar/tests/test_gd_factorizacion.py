"""Gradiente de la SCE y GD matricial (informe 3.4/6)."""

import logging

import numpy as np
import pytest

from src.errores import DivergenciaError, LambdaNegativoError
from src.gradiente import entrenar_gd, gradiente_sce
from src.modelo import f_regularizada, inicializar_factores, sce


def _gradiente_numerico(f, X, h=1e-6):
    """Aproxima el gradiente de f en X por diferencias finitas centradas."""
    grad = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_mas = X.copy()
            X_mas[i, j] += h
            X_menos = X.copy()
            X_menos[i, j] -= h
            grad[i, j] = (f(X_mas) - f(X_menos)) / (2 * h)
    return grad


def test_gradiente_sce_coincide_con_diferencias_finitas(matriz_pequena_aleatoria, generador_fijo):
    R, M = matriz_pequena_aleatoria
    m, n = R.shape
    k = 2
    U = generador_fijo.uniform(0.0, 1.0, size=(m, k))
    V = generador_fijo.uniform(0.0, 1.0, size=(n, k))

    grad_U, grad_V = gradiente_sce(R, M, U, V)

    grad_U_numerico = _gradiente_numerico(lambda U_: sce(R, M, U_, V), U)
    grad_V_numerico = _gradiente_numerico(lambda V_: sce(R, M, U, V_), V)

    error_relativo_U = np.linalg.norm(grad_U - grad_U_numerico) / np.linalg.norm(grad_U_numerico)
    error_relativo_V = np.linalg.norm(grad_V - grad_V_numerico) / np.linalg.norm(grad_V_numerico)

    assert error_relativo_U < 1e-5
    assert error_relativo_V < 1e-5


def test_iteracion_gd_es_simultanea_contra_referencia(matriz_pequena_aleatoria, generador_fijo):
    R, M = matriz_pequena_aleatoria
    m, n = R.shape
    k = 2
    U0 = generador_fijo.uniform(0.0, 1.0, size=(m, k))
    V0 = generador_fijo.uniform(0.0, 1.0, size=(n, k))
    eta = 0.01

    # max_iter=1 fuerza exactamente una iteración, sin importar el epsilon.
    resultado = entrenar_gd(R, M, U0.copy(), V0.copy(), eta=eta, epsilon=1e-12, max_iter=1)

    # Referencia simultánea: ambos gradientes con (U0, V0), calculados en el
    # propio test.
    grad_U_ref, grad_V_ref = gradiente_sce(R, M, U0, V0)
    U_esperado = U0 - eta * grad_U_ref
    V_esperado = V0 - eta * grad_V_ref

    np.testing.assert_allclose(resultado.U, U_esperado)
    np.testing.assert_allclose(resultado.V, V_esperado)

    # Si fuera alternado (actualizar U y recién ahí calcular el gradiente de
    # V con la U ya actualizada), V daría distinto de lo simultáneo.
    _, grad_V_alternado = gradiente_sce(R, M, U_esperado, V0)
    V_alternado = V0 - eta * grad_V_alternado
    assert not np.allclose(resultado.V, V_alternado)


def test_entrenar_gd_lanza_divergencia_si_f_no_es_finito(matriz_pequena_aleatoria, generador_fijo):
    R, M = matriz_pequena_aleatoria
    m, n = R.shape
    k = 2
    U0 = generador_fijo.uniform(0.0, 1.0, size=(m, k))
    V0 = generador_fijo.uniform(0.0, 1.0, size=(n, k))

    # eta enorme: tiene que divergir (f no finito) en pocas iteraciones, no
    # correr hasta max_iter.
    with pytest.raises(DivergenciaError) as exc_info:
        entrenar_gd(R, M, U0, V0, eta=1e10, epsilon=1e-8, max_iter=1000)

    assert exc_info.value.iteracion < 5
    assert exc_info.value.eta == 1e10


def test_entrenar_gd_loguea_debug_por_iteracion_e_info_cada_100_y_resumen(
    matriz_ejemplo_informe, generador_fijo, caplog
):
    R, M = matriz_ejemplo_informe
    U0, V0 = inicializar_factores(m=4, n=5, k=2, escala=1.0, generador=generador_fijo)

    # epsilon=0.0 nunca corta por tolerancia: fuerza exactamente max_iter
    # iteraciones, para poder contar los logs. eta=0.01 ya se usó en otros
    # tests sobre esta misma matriz sin diverger.
    with caplog.at_level(logging.DEBUG, logger="src.gradiente"):
        resultado = entrenar_gd(R, M, U0, V0, eta=0.01, epsilon=0.0, max_iter=250)

    assert resultado.n_iteraciones == 250

    mensajes_debug = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG and "GD iteración" in r.getMessage()
    ]
    mensajes_info_iteracion = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "GD iteración" in r.getMessage()
    ]
    mensajes_resumen = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "terminó" in r.getMessage()
    ]

    assert len(mensajes_debug) == 250
    assert len(mensajes_info_iteracion) == 2
    assert len(mensajes_resumen) == 1


# --- λ en gradiente_sce ---


def test_gradiente_con_lambda_coincide_con_diferencias_finitas(
    matriz_pequena_aleatoria, generador_fijo
):
    # Mismo método que el test del gradiente de la SCE, contra diferencias finitas de f_regularizada.
    R, M = matriz_pequena_aleatoria
    m, n = R.shape
    k = 2
    U = generador_fijo.uniform(0.0, 1.0, size=(m, k))
    V = generador_fijo.uniform(0.0, 1.0, size=(n, k))
    lambda_ = 0.8

    grad_U, grad_V = gradiente_sce(R, M, U, V, lambda_=lambda_)

    grad_U_numerico = _gradiente_numerico(lambda U_: f_regularizada(R, M, U_, V, lambda_), U)
    grad_V_numerico = _gradiente_numerico(lambda V_: f_regularizada(R, M, U, V_, lambda_), V)

    error_relativo_U = np.linalg.norm(grad_U - grad_U_numerico) / np.linalg.norm(grad_U_numerico)
    error_relativo_V = np.linalg.norm(grad_V - grad_V_numerico) / np.linalg.norm(grad_V_numerico)

    assert error_relativo_U < 1e-5
    assert error_relativo_V < 1e-5
    # con λ > 0 el gradiente no es el de la SCE pura
    grad_U_sin_lambda, _ = gradiente_sce(R, M, U, V)
    assert not np.allclose(grad_U, grad_U_sin_lambda)


def test_gradiente_con_lambda_cero_es_identico_al_actual(matriz_pequena_aleatoria, generador_fijo):
    R, M = matriz_pequena_aleatoria
    m, n = R.shape
    U = generador_fijo.uniform(0.0, 1.0, size=(m, 2))
    V = generador_fijo.uniform(0.0, 1.0, size=(n, 2))

    grad_U, grad_V = gradiente_sce(R, M, U, V, lambda_=0.0)
    grad_U_actual, grad_V_actual = gradiente_sce(R, M, U, V)

    # igualdad exacta, no aproximada
    assert np.array_equal(grad_U, grad_U_actual)
    assert np.array_equal(grad_V, grad_V_actual)


def test_gradiente_con_lambda_negativo_lanza_lambda_negativo_error(matriz_pequena_aleatoria):
    R, M = matriz_pequena_aleatoria
    m, n = R.shape

    with pytest.raises(LambdaNegativoError, match=r"-1"):
        gradiente_sce(R, M, np.ones((m, 2)), np.ones((n, 2)), lambda_=-1.0)
