"""Experimento 4 (experimentos/bitacora.md): ordenar por la estimación relativa."""

import numpy as np
import pytest

from experimentos import concentracion
from experimentos.ordenamiento_relativo import (
    medir_tops,
    puntajes_relativos,
    simular_ambos_ordenamientos,
)
from src.als import entrenar_als
from src.modelo import inicializar_factores, predecir


def test_puntaje_relativo_es_r_hat_menos_el_promedio_de_la_pelicula_sobre_los_usuarios():
    generador = np.random.default_rng(4)
    U = generador.normal(size=(6, 2))
    V = generador.normal(size=(5, 2))

    R_hat = predecir(U, V)
    esperado = R_hat - R_hat.mean(axis=0, keepdims=True)  # r̂ᵢⱼ − promedio_i r̂ᵢⱼ

    np.testing.assert_allclose(puntajes_relativos(U, V), esperado)


def test_medir_tops_excluye_lo_calificado_y_promedia_r_hat_absoluto_de_lo_recomendado():
    # Puntajes que ordenan distinto que r̂, para ver que el promedio usa r̂.
    puntajes = np.array([[1.0, 3.0, 2.0], [3.0, 2.0, 1.0]])
    R_hat = np.array([[4.0, 5.0, 2.0], [1.0, 3.0, 4.5]])
    M = np.array([[False, True, False], [False, False, False]])
    titulos = {0: "A", 1: "B", 2: "C"}

    resultado = medir_tops(puntajes, R_hat, M, titulos, top_n=1, umbral_frecuente=0.2)

    # usuario 0: B está calificada → recomienda C (r̂ = 2); usuario 1: A (r̂ = 1)
    assert resultado.distintas_en_top_n == 2
    assert resultado.frecuencia_mas_frecuente == 0.5
    assert resultado.promedio_r_hat == pytest.approx((2.0 + 1.0) / 2)
    assert resultado.n_frecuentes == 2  # las dos aparecen en el 50 % (> 20 %)


def test_simulacion_absoluta_coincide_con_concentracion_simular():
    generador = np.random.default_rng(2)
    m, n = 15, 20
    M = generador.uniform(size=(m, n)) < 0.7
    R = np.where(M, generador.integers(1, 6, size=(m, n)).astype(float), np.nan)
    titulos = {j: f"película {j}" for j in range(n)}
    notas = np.array([1.0, 3.0, 4.0, 5.0, 5.0])
    parametros = dict(
        n_usuarios=200, min_calificadas=5, max_calificadas=10, n_a_calificar=12,
        top_n=5, semilla_simulacion=0,
    )

    referencia = concentracion.simular(
        R, M, titulos, centrar=False, notas=notas, k=2, epsilon=1e-6, max_iter=200,
        semilla=7, umbral_frecuente=0.2, **parametros,
    )
    U0, V0 = inicializar_factores(m=m, n=n, k=2, escala=1.0, generador=np.random.default_rng(7))
    modelo = entrenar_als(R, M, U0, V0, epsilon=1e-6, max_iter=200)

    absoluto, _ = simular_ambos_ordenamientos(
        modelo.U, modelo.V, M, titulos, notas, umbral_frecuente=0.2, **parametros
    )

    assert absoluto.distintas_en_top_n == referencia.distintas_en_top_n
    assert absoluto.frecuencia_mas_frecuente == referencia.frecuencia_mas_frecuente
    assert absoluto.n_frecuentes == referencia.n_frecuentes
