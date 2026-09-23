"""Gradiente de la SCE (informe 3.4/6)."""

import numpy as np

from src.gradiente import gradiente_sce
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
