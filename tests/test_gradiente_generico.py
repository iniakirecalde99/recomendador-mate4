"""Descenso de gradiente genérico (informe 3.4)."""

import numpy as np

from src.gradiente import descenso_gradiente


def test_descenso_gradiente_converge_al_minimo_conocido():
    # f(x,y) = (x-1)^2 + (y-2)^2 + (x+y-6)^2 (informe 3.4, tabla 2).
    # Mínimo analítico en (2, 3), con f = 3.
    def f(x):
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[0] + x[1] - 6) ** 2

    def grad_f(x):
        return np.array(
            [
                4 * x[0] + 2 * x[1] - 14,
                2 * x[0] + 4 * x[1] - 16,
            ]
        )

    resultado = descenso_gradiente(
        f=f,
        grad_f=grad_f,
        x0=np.array([0.0, 0.0]),
        eta=0.1,
        epsilon=1e-10,
        max_iter=100_000,
    )

    assert resultado.motivo_corte == "tolerancia"
    np.testing.assert_allclose(resultado.x, [2.0, 3.0], atol=1e-4)
    assert abs(f(resultado.x) - 3.0) < 1e-4
    assert resultado.historial_f[0] == f(np.array([0.0, 0.0]))
    assert len(resultado.historial_f) == resultado.n_iteraciones + 1
