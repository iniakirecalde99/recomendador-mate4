"""Carga, filtro, títulos y orquestación de MovieLens."""

from pathlib import Path

import numpy as np
import pytest

from src.datos import (
    ConjuntoCalificaciones,
    cargar_calificaciones,
    cargar_generos,
    cargar_titulos,
    construir_indice_a_titulo,
    filtrar_por_minimo,
    particionar,
    preparar_datos_movielens,
)
from src.errores import ErrorDatosInsuficientes, UmbralInsuficienteError

RUTA_MOVIELENS = Path(__file__).resolve().parent.parent / "data" / "ml-latest-small"


def test_cargar_calificaciones_reindexa_ids_y_arma_mascara(tmp_path):
    contenido = (
        "userId,movieId,rating,timestamp\n"
        "3,7,5.0,888\n"
        "3,2,4.5,889\n"
        "9,7,1.0,890\n"
    )
    ruta_archivo = tmp_path / "ratings.csv"
    ruta_archivo.write_text(contenido, encoding="utf-8")

    resultado = cargar_calificaciones(ruta_archivo)

    assert resultado.id_usuario_a_indice == {3: 0, 9: 1}
    assert resultado.id_pelicula_a_indice == {2: 0, 7: 1}
    assert resultado.R.shape == (2, 2)
    assert resultado.M.dtype == bool
    assert resultado.M.sum() == 3

    np.testing.assert_allclose(resultado.R[0, 0], 4.5)  # usuario 3, película 2 (media estrella)
    np.testing.assert_allclose(resultado.R[0, 1], 5.0)  # usuario 3, película 7
    np.testing.assert_allclose(resultado.R[1, 1], 1.0)  # usuario 9, película 7
    assert np.isnan(resultado.R[1, 0])  # usuario 9, película 2: sin calificar

    assert resultado.M.tolist() == [[True, True], [False, True]]


@pytest.mark.movielens
def test_cargar_calificaciones_dimensiones_y_mascara():
    resultado = cargar_calificaciones(RUTA_MOVIELENS / "ratings.csv")

    # 9724 películas con al menos una calificación (movies.csv lista 9742)
    assert resultado.R.shape == (610, 9724)
    assert resultado.M.sum() == 100_836


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

    resultado = filtrar_por_minimo(datos, umbral=2, k=2)

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
        filtrar_por_minimo(datos, umbral=2, k=2)


def test_filtrar_por_minimo_lanza_umbral_insuficiente_si_umbral_es_menor_que_k():
    R = np.array([[5.0, 4.0], [3.0, 2.0]])
    M = np.array([[True, True], [True, True]])
    datos = ConjuntoCalificaciones(
        R=R, M=M, id_usuario_a_indice={1: 0, 2: 1}, id_pelicula_a_indice={1: 0, 2: 1},
    )

    with pytest.raises(UmbralInsuficienteError) as exc_info:
        filtrar_por_minimo(datos, umbral=1, k=2)

    assert exc_info.value.umbral == 1
    assert exc_info.value.k == 2


def test_cargar_titulos_y_construir_indice_a_titulo(tmp_path):
    contenido = (
        "movieId,title,genres\n"
        "1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy\n"
        '11,"American President, The (1995)",Comedy|Drama|Romance\n'
        "3,Am\xe9lie (2001),Comedy|Romance\n"
    )
    ruta_archivo = tmp_path / "movies.csv"
    ruta_archivo.write_text(contenido, encoding="utf-8")

    titulos_por_id = cargar_titulos(ruta_archivo)

    assert titulos_por_id == {
        1: "Toy Story (1995)",
        11: "American President, The (1995)",  # título entre comillas, con coma
        3: "Am\xe9lie (2001)",
    }

    # id_pelicula_a_indice típicamente sale de cargar_calificaciones/filtrar_por_minimo:
    # solo cubre las películas que sobrevivieron al filtro (acá, 1 y 3; la 2 quedó afuera).
    id_pelicula_a_indice = {1: 0, 3: 1}  # la 11 quedó afuera
    indice_a_titulo = construir_indice_a_titulo(id_pelicula_a_indice, titulos_por_id)

    assert indice_a_titulo == {0: "Toy Story (1995)", 1: "Am\xe9lie (2001)"}


def test_cargar_generos_separa_la_columna_genres_de_movies_csv(tmp_path):
    contenido = (
        "movieId,title,genres\n"
        "1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy\n"
        '11,"American President, The (1995)",Comedy|Drama|Romance\n'
        "114335,La cravate (1957),(no genres listed)\n"
    )
    ruta_archivo = tmp_path / "movies.csv"
    ruta_archivo.write_text(contenido, encoding="utf-8")

    generos_por_id = cargar_generos(ruta_archivo)

    assert generos_por_id == {
        1: ["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
        11: ["Comedy", "Drama", "Romance"],
        114335: ["(no genres listed)"],
    }


def test_preparar_datos_movielens_integra_carga_filtro_y_titulos(tmp_path):
    # k=2: la película 3 tiene una sola calificación (de usuario 1) y el
    # filtro la elimina; usuarios 1 y 2 quedan con 2 calificaciones cada uno
    # (películas 1 y 2), así que ninguno se elimina en una pasada posterior.
    ruta_ratings = tmp_path / "ratings.csv"
    ruta_ratings.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,1,5.0,100\n"
        "1,2,4.0,101\n"
        "1,3,3.0,102\n"
        "2,1,3.0,103\n"
        "2,2,2.0,104\n",
        encoding="utf-8",
    )
    ruta_movies = tmp_path / "movies.csv"
    ruta_movies.write_text(
        "movieId,title,genres\n"
        "1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy\n"
        "2,GoldenEye (1995),Action|Adventure|Thriller\n"
        "3,Nixon (1995),Drama\n",
        encoding="utf-8",
    )

    datos = preparar_datos_movielens(ruta_ratings, ruta_movies, umbral=2, k=2)

    # Película 3 filtrada; usuarios 1 y 2, y películas 1 y 2 sobreviven.
    assert set(datos.calificaciones.id_usuario_a_indice) == {1, 2}
    assert set(datos.calificaciones.id_pelicula_a_indice) == {1, 2}
    assert datos.calificaciones.R.shape == (2, 2)

    # Los títulos están indexados por el índice FINAL (post-filtro), no por
    # el id crudo; la película 3 (filtrada) no debe aparecer.
    idx_pelicula1 = datos.calificaciones.id_pelicula_a_indice[1]
    idx_pelicula2 = datos.calificaciones.id_pelicula_a_indice[2]
    assert datos.titulos_por_indice[idx_pelicula1] == "Toy Story (1995)"
    assert datos.titulos_por_indice[idx_pelicula2] == "GoldenEye (1995)"
    assert len(datos.titulos_por_indice) == 2


# --- Partición entrenamiento/prueba ---


def _mascara_aleatoria(semilla, m=30, n=40, densidad=0.3):
    return np.random.default_rng(semilla).uniform(size=(m, n)) < densidad


def test_particion_es_disjunta_y_su_union_es_omega():
    M = _mascara_aleatoria(1)

    M_ent, M_prueba = particionar(M, fraccion_prueba=0.2, generador=np.random.default_rng(5))

    assert not np.any(M_ent & M_prueba)
    assert np.array_equal(M_ent | M_prueba, M)
    assert M_prueba.sum() > 0
    # a lo sumo la fracción pedida (puede ser menos si hubo que pasar pares)
    assert M_prueba.sum() <= round(0.2 * M.sum())


def test_particion_es_reproducible_con_la_semilla():
    M = _mascara_aleatoria(2)

    a = particionar(M, fraccion_prueba=0.2, generador=np.random.default_rng(9))
    b = particionar(M, fraccion_prueba=0.2, generador=np.random.default_rng(9))
    c = particionar(M, fraccion_prueba=0.2, generador=np.random.default_rng(10))

    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_particion_todo_usuario_y_pelicula_de_prueba_tiene_calificaciones_de_entrenamiento():
    M = _mascara_aleatoria(3, densidad=0.1)  # rala: muchas filas/columnas con pocas calificaciones

    M_ent, M_prueba = particionar(M, fraccion_prueba=0.5, generador=np.random.default_rng(4))

    filas_prueba, columnas_prueba = np.nonzero(M_prueba)
    assert np.all(M_ent[filas_prueba, :].any(axis=1))
    assert np.all(M_ent[:, columnas_prueba].any(axis=0))


def test_particion_pasa_a_entrenamiento_los_pares_que_dejarian_huerfano_a_un_usuario_o_pelicula():
    # Con fracción 1 todo Ω se sortea a prueba: todos los usuarios y películas
    # quedan sin entrenamiento, así que todos los pares vuelven a entrenamiento.
    M = np.array([[True, True, False], [True, False, True]])

    M_ent, M_prueba = particionar(M, fraccion_prueba=1.0, generador=np.random.default_rng(0))

    assert np.array_equal(M_ent, M)
    assert not M_prueba.any()
