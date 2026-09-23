"""Reproducibilidad de ALS y GD con la misma semilla."""

import numpy as np

from src.als import entrenar_als
from src.gradiente import entrenar_gd
from src.modelo import inicializar_factores


def test_misma_semilla_misma_ejecucion_als(matriz_ejemplo_informe):
    R, M = matriz_ejemplo_informe

    U0_a, V0_a = inicializar_factores(m=4, n=5, k=2, escala=1.0, generador=np.random.default_rng(123))
    U0_b, V0_b = inicializar_factores(m=4, n=5, k=2, escala=1.0, generador=np.random.default_rng(123))

    resultado_a = entrenar_als(R, M, U0_a, V0_a, epsilon=1e-8, max_iter=100)
    resultado_b = entrenar_als(R, M, U0_b, V0_b, epsilon=1e-8, max_iter=100)

    np.testing.assert_array_equal(resultado_a.U, resultado_b.U)
    np.testing.assert_array_equal(resultado_a.V, resultado_b.V)
    assert resultado_a.historial_f == resultado_b.historial_f
    assert resultado_a.n_iteraciones == resultado_b.n_iteraciones
    assert resultado_a.motivo_corte == resultado_b.motivo_corte


def test_misma_semilla_misma_ejecucion_gd(matriz_ejemplo_informe):
    R, M = matriz_ejemplo_informe

    U0_a, V0_a = inicializar_factores(m=4, n=5, k=2, escala=1.0, generador=np.random.default_rng(123))
    U0_b, V0_b = inicializar_factores(m=4, n=5, k=2, escala=1.0, generador=np.random.default_rng(123))

    resultado_a = entrenar_gd(R, M, U0_a, V0_a, eta=0.01, epsilon=1e-8, max_iter=300)
    resultado_b = entrenar_gd(R, M, U0_b, V0_b, eta=0.01, epsilon=1e-8, max_iter=300)

    np.testing.assert_array_equal(resultado_a.U, resultado_b.U)
    np.testing.assert_array_equal(resultado_a.V, resultado_b.V)
    assert resultado_a.historial_f == resultado_b.historial_f
    assert resultado_a.n_iteraciones == resultado_b.n_iteraciones
    assert resultado_a.motivo_corte == resultado_b.motivo_corte
