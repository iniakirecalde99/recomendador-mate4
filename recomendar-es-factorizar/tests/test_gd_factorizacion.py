"""Gradiente de la SCE y GD matricial (informe 3.4/6)."""

import numpy as np

from src.gradiente import entrenar_gd, gradiente_sce
from src.modelo import sce


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
