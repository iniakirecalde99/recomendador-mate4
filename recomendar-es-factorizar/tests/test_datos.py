"""Carga, filtro por mínimo y títulos de MovieLens."""

from pathlib import Path

import numpy as np
import pytest

from src.datos import (
    ConjuntoCalificaciones,
    cargar_calificaciones,
    cargar_titulos,
    construir_indice_a_titulo,
    filtrar_por_minimo,
)
from src.errores import ErrorDatosInsuficientes

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


def test_filtrar_por_minimo_todas_las_filas_y_columnas_tienen_al_menos_k():
    # Cascada de tres pasadas para k=2 (usuarios U1,U2,U3; películas M1..M5):
    #   pasada 1: M4 tiene 1 calificación (solo U3) -> se elimina.
    #   pasada 2: sin M4, U3 queda con 1 calificación (solo M5) -> se elimina.
    #   pasada 3: sin U3, M5 queda con 1 calificación (solo U1) -> se elimina.
    #   pasada 4: {U1,U2} x {M1,M2,M3} ya no tiene nada para eliminar.
    # Cualquier filtro de una sola pasada (en cualquier orden filas/columnas)
    # se queda a mitad de camino y no llega a este resultado.
    R = np.full((3, 5), np.nan)
    M = np.zeros((3, 5), dtype=bool)
    calificaciones = {
        (0, 0): 5.0, (0, 1): 4.0, (0, 2): 3.0, (0, 4): 2.0,  # U1: M1,M2,M3,M5
        (1, 0): 1.0, (1, 1): 2.0, (1, 2): 3.0,  # U2: M1,M2,M3
        (2, 3): 4.0, (2, 4): 5.0,  # U3: M4,M5
    }
    for (i, j), calificacion in calificaciones.items():
        R[i, j] = calificacion
        M[i, j] = True

    datos = ConjuntoCalificaciones(
        R=R,
        M=M,
        id_usuario_a_indice={1: 0, 2: 1, 3: 2},
        id_pelicula_a_indice={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    )

    resultado = filtrar_por_minimo(datos, k=2)

    assert resultado.id_usuario_a_indice == {1: 0, 2: 1}
    assert resultado.id_pelicula_a_indice == {1: 0, 2: 1, 3: 2}
    assert resultado.R.shape == (2, 3)
    assert (resultado.M.sum(axis=0) >= 2).all()
    assert (resultado.M.sum(axis=1) >= 2).all()

    # al menos dos calificaciones concretas, en la posición que indica el
    # mapeo NUEVO que devuelve filtrar_por_minimo (no el viejo de datos).
    idx_u1 = resultado.id_usuario_a_indice[1]
    idx_u2 = resultado.id_usuario_a_indice[2]
    idx_m2 = resultado.id_pelicula_a_indice[2]
    idx_m3 = resultado.id_pelicula_a_indice[3]
    assert resultado.R[idx_u1, idx_m3] == 3.0  # U1 - M3
    assert resultado.R[idx_u2, idx_m2] == 2.0  # U2 - M2


def test_filtrar_por_minimo_lanza_error_si_vacia_la_matriz():
    R = np.array([[5.0]])
    M = np.array([[True]])
    datos = ConjuntoCalificaciones(
        R=R, M=M, id_usuario_a_indice={1: 0}, id_pelicula_a_indice={1: 0},
    )

    with pytest.raises(ErrorDatosInsuficientes):
        filtrar_por_minimo(datos, k=2)


def test_cargar_titulos_y_construir_indice_a_titulo(tmp_path):
    contenido = (
        "1|Toy Story (1995)|01-Jan-1995||url1|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
        "2|GoldenEye (1995)|01-Jan-1995||url2|0|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
        "3|Am\xe9lie (2001)|01-Jan-2001||url3|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
    )
    ruta_archivo = tmp_path / "u.item"
    ruta_archivo.write_bytes(contenido.encode("latin-1"))

    titulos_por_id = cargar_titulos(ruta_archivo)

    assert titulos_por_id == {
        1: "Toy Story (1995)",
        2: "GoldenEye (1995)",
        3: "Am\xe9lie (2001)",
    }

    # id_pelicula_a_indice típicamente sale de cargar_calificaciones/filtrar_por_minimo:
    # solo cubre las películas que sobrevivieron al filtro (acá, 1 y 3; la 2 quedó afuera).
    id_pelicula_a_indice = {1: 0, 3: 1}
    indice_a_titulo = construir_indice_a_titulo(id_pelicula_a_indice, titulos_por_id)

    assert indice_a_titulo == {0: "Toy Story (1995)", 1: "Am\xe9lie (2001)"}
