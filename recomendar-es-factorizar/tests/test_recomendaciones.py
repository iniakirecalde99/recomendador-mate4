"""Top-N y extremos por factor latente."""

import numpy as np

from src.recomendaciones import extremos_por_factor, recomendar_top_n


def test_top_n_excluye_peliculas_ya_calificadas():
    U = np.array([[1.0, 0.0]])  # 1 usuario
    V = np.array(
        [
            [5.0, 0.0],  # película 0: r_hat=5, ya calificada
            [4.0, 0.0],  # película 1: r_hat=4
            [3.0, 0.0],  # película 2: r_hat=3
            [2.0, 0.0],  # película 3: r_hat=2
        ]
    )
    M = np.array([[True, False, False, False]])
    titulos_por_indice = {0: "Ya calificada", 1: "Segunda", 2: "Tercera", 3: "Cuarta"}

    recomendaciones = recomendar_top_n(U, V, M, indice_usuario=0, n=2, titulos_por_indice=titulos_por_indice)

    titulos_recomendados = [titulo for titulo, _r_hat in recomendaciones]
    assert "Ya calificada" not in titulos_recomendados
    assert titulos_recomendados == ["Segunda", "Tercera"]


def test_extremos_por_factor_devuelve_la_cantidad_pedida_por_columna():
    V = np.array(
        [
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0],
            [6.0, 0.0],
        ]
    )
    titulos_por_indice = {i: f"Película {i}" for i in range(6)}

    resultado = extremos_por_factor(V, titulos_por_indice, cantidad=2)

    assert len(resultado) == 2  # 2 factores latentes (columnas de V)

    factor0, mayores0, menores0 = resultado[0]
    assert factor0 == 0
    assert [titulo for titulo, _valor in mayores0] == ["Película 5", "Película 4"]
    assert [titulo for titulo, _valor in menores0] == ["Película 0", "Película 1"]

    factor1, mayores1, menores1 = resultado[1]
    assert factor1 == 1
    assert [titulo for titulo, _valor in mayores1] == ["Película 0", "Película 1"]
    assert [titulo for titulo, _valor in menores1] == ["Película 5", "Película 4"]
