"""Experimento 3 (experimentos/bitacora.md): concentración del top-10 de los usuarios reales."""

import numpy as np

from experimentos.usuarios_reales import concentracion_usuarios_reales


def test_top_n_de_cada_usuario_excluye_lo_calificado_y_cuenta_apariciones():
    # U·Vᵀ = [[4, 3, 2, 1], [1, 2, 3, 4], [4, 3, 2, 1]]
    U = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    V = np.array([[4.0, 1.0], [3.0, 2.0], [2.0, 3.0], [1.0, 4.0]])
    M = np.array(
        [
            [True, False, False, False],  # ya calificó la 0: su top-2 es 1, 2
            [False, False, False, True],  # ya calificó la 3: su top-2 es 2, 1
            [False, False, False, False],  # nada calificado: su top-2 es 0, 1
        ]
    )
    titulos = {0: "A", 1: "B", 2: "C", 3: "D"}

    resultado = concentracion_usuarios_reales(U, V, M, titulos, top_n=2, n_mas_frecuentes=3)

    assert resultado.n_usuarios == 3
    assert resultado.distintas_en_top_n == 3  # A, B y C; D nunca entra
    assert resultado.mas_frecuente == "B"  # en los tres top-2
    assert resultado.frecuencia_mas_frecuente == 1.0
    assert resultado.mas_frecuentes == [("B", 1.0), ("C", 2 / 3), ("A", 1 / 3)]
