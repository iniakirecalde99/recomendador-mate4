"""Carga de calificaciones crudas."""

from pathlib import Path

import numpy as np
import pytest

from src.datos import cargar_calificaciones

RUTA_MOVIELENS = Path(__file__).resolve().parent.parent / "data" / "ml-100k"


def test_cargar_calificaciones_reindexa_ids_y_arma_mascara(tmp_path):
    contenido = (
        "3\t7\t5\t888\n"
        "3\t2\t4\t889\n"
        "9\t7\t1\t890\n"
    )
    ruta_archivo = tmp_path / "u.data"
    ruta_archivo.write_text(contenido, encoding="utf-8")

    resultado = cargar_calificaciones(ruta_archivo)

    assert resultado.id_usuario_a_indice == {3: 0, 9: 1}
    assert resultado.id_pelicula_a_indice == {2: 0, 7: 1}
    assert resultado.R.shape == (2, 2)
    assert resultado.M.dtype == bool
    assert resultado.M.sum() == 3

    np.testing.assert_allclose(resultado.R[0, 0], 4.0)  # usuario 3, película 2
    np.testing.assert_allclose(resultado.R[0, 1], 5.0)  # usuario 3, película 7
    np.testing.assert_allclose(resultado.R[1, 1], 1.0)  # usuario 9, película 7
    assert np.isnan(resultado.R[1, 0])  # usuario 9, película 2: sin calificar

    assert resultado.M.tolist() == [[True, True], [False, True]]


@pytest.mark.movielens
def test_cargar_calificaciones_dimensiones_y_mascara():
    resultado = cargar_calificaciones(RUTA_MOVIELENS / "u.data")

    assert resultado.R.shape == (943, 1682)
    assert resultado.M.sum() == 100_000
