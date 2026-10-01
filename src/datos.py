"""Carga y preprocesamiento de MovieLens."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.errores import ErrorDatosInsuficientes, UmbralInsuficienteError

LOGGER = logging.getLogger(__name__)


@dataclass
class ConjuntoCalificaciones:
    """Matriz de calificaciones reindexada desde 0.

    Atributos:
        R: matriz m × n de calificaciones, con NaN en los huecos.
        M: máscara booleana m × n, True donde (i, j) ∈ Ω.
        id_usuario_a_indice: id de usuario tal como aparece en el archivo
            crudo -> índice de fila en R/M.
        id_pelicula_a_indice: id de película tal como aparece en el archivo
            crudo -> índice de columna en R/M.
    """

    R: np.ndarray
    M: np.ndarray
    id_usuario_a_indice: dict[int, int]
    id_pelicula_a_indice: dict[int, int]


def cargar_calificaciones(ruta_archivo: Path) -> ConjuntoCalificaciones:
    """Carga `ratings.csv` de MovieLens.

    CSV con encabezado `userId,movieId,rating,timestamp`, una calificación
    por línea (de 0.5 a 5, de a media estrella). Reindexa los ids de usuario
    y de película desde 0 (según su orden ascendente) y arma R (con NaN en
    los huecos) y M. No aplica ningún filtro.
    """
    filas: list[tuple[int, int, float]] = []
    with Path(ruta_archivo).open(encoding="utf-8", newline="") as archivo:
        for registro in csv.DictReader(archivo):
            filas.append(
                (int(registro["userId"]), int(registro["movieId"]), float(registro["rating"]))
            )

    ids_usuario = sorted({id_usuario for id_usuario, _id_pelicula, _calificacion in filas})
    ids_pelicula = sorted({id_pelicula for _id_usuario, id_pelicula, _calificacion in filas})
    id_usuario_a_indice = {id_crudo: indice for indice, id_crudo in enumerate(ids_usuario)}
    id_pelicula_a_indice = {id_crudo: indice for indice, id_crudo in enumerate(ids_pelicula)}

    m = len(ids_usuario)
    n = len(ids_pelicula)
    R = np.full((m, n), np.nan, dtype=float)
    M = np.zeros((m, n), dtype=bool)

    for id_usuario, id_pelicula, calificacion in filas:
        i = id_usuario_a_indice[id_usuario]
        j = id_pelicula_a_indice[id_pelicula]
        R[i, j] = calificacion
        M[i, j] = True

    LOGGER.info(
        "cargadas %d calificaciones de %d usuarios y %d películas desde %s",
        len(filas), m, n, ruta_archivo,
    )

    return ConjuntoCalificaciones(
        R=R,
        M=M,
        id_usuario_a_indice=id_usuario_a_indice,
        id_pelicula_a_indice=id_pelicula_a_indice,
    )


def filtrar_por_minimo(datos: ConjuntoCalificaciones, umbral: int, k: int) -> ConjuntoCalificaciones:
    """Elimina filas y columnas con menos de `umbral` calificaciones.

    `umbral` es un parámetro propio del filtro, independiente de `k` (la
    dimensión latente), pero tiene que ser >= k: el sistema de cada fila en
    ALS es k×k y necesita al menos k observaciones para tener solución
    única — pero tener exactamente k observaciones no alcanza para que esa
    solución sea razonable (la calibración con MovieLens real mostró
    `SistemaSingularError` y predicciones fuera de Ω con magnitud absurda
    incluso con umbral == k; ver
    experimentos/bitacora.md). Lanza `UmbralInsuficienteError` si `umbral` < `k`.

    Repite la eliminación hasta que una pasada no elimine nada: sacar una
    columna puede dejar a una fila por debajo del umbral (o viceversa), así
    que una sola pasada no alcanza. Reconstruye `id_usuario_a_indice` y
    `id_pelicula_a_indice` desde cero para los ids que sobrevivieron, con
    índices nuevos y contiguos desde 0 (no reutiliza los índices viejos de
    `datos`, que quedan obsoletos apenas se elimina la primera fila o
    columna).

    Lanza `ErrorDatosInsuficientes` si el filtro deja la matriz sin filas o
    sin columnas.
    """
    if umbral < k:
        raise UmbralInsuficienteError(umbral, k)

    indice_a_id_usuario = {indice: id_crudo for id_crudo, indice in datos.id_usuario_a_indice.items()}
    indice_a_id_pelicula = {indice: id_crudo for id_crudo, indice in datos.id_pelicula_a_indice.items()}

    filas_activas = np.arange(datos.M.shape[0])
    columnas_activas = np.arange(datos.M.shape[1])
    usuarios_eliminados = 0
    peliculas_eliminadas = 0
    n_pasada = 0

    while True:
        n_pasada += 1
        submascara = datos.M[np.ix_(filas_activas, columnas_activas)]
        conteo_filas = submascara.sum(axis=1)
        conteo_columnas = submascara.sum(axis=0)

        filas_a_mantener = filas_activas[conteo_filas >= umbral]
        columnas_a_mantener = columnas_activas[conteo_columnas >= umbral]

        if len(filas_a_mantener) == len(filas_activas) and len(columnas_a_mantener) == len(
            columnas_activas
        ):
            break

        usuarios_eliminados += len(filas_activas) - len(filas_a_mantener)
        peliculas_eliminadas += len(columnas_activas) - len(columnas_a_mantener)
        LOGGER.debug(
            "filtro por umbral=%d, pasada %d: quedan %d usuarios y %d películas",
            umbral, n_pasada, len(filas_a_mantener), len(columnas_a_mantener),
        )

        if len(filas_a_mantener) == 0 or len(columnas_a_mantener) == 0:
            raise ErrorDatosInsuficientes(umbral)

        filas_activas = filas_a_mantener
        columnas_activas = columnas_a_mantener

    R_filtrado = datos.R[np.ix_(filas_activas, columnas_activas)]
    M_filtrado = datos.M[np.ix_(filas_activas, columnas_activas)]

    nuevo_id_usuario_a_indice = {
        indice_a_id_usuario[indice_viejo]: nuevo_indice
        for nuevo_indice, indice_viejo in enumerate(filas_activas)
    }
    nuevo_id_pelicula_a_indice = {
        indice_a_id_pelicula[indice_viejo]: nuevo_indice
        for nuevo_indice, indice_viejo in enumerate(columnas_activas)
    }

    calificaciones_eliminadas = int(datos.M.sum() - M_filtrado.sum())
    if usuarios_eliminados or peliculas_eliminadas:
        LOGGER.warning(
            "filtro por umbral=%d: se eliminaron %d usuarios, %d películas y %d calificaciones",
            umbral, usuarios_eliminados, peliculas_eliminadas, calificaciones_eliminadas,
        )

    return ConjuntoCalificaciones(
        R=R_filtrado,
        M=M_filtrado,
        id_usuario_a_indice=nuevo_id_usuario_a_indice,
        id_pelicula_a_indice=nuevo_id_pelicula_a_indice,
    )


def cargar_titulos(ruta_movies: Path) -> dict[int, str]:
    """Carga `movies.csv` y devuelve el mapeo id de película crudo → título.

    CSV en UTF-8 con encabezado `movieId,title,genres`; los títulos con coma
    vienen entre comillas, por eso se lee con el módulo `csv`.
    """
    titulos_por_id: dict[int, str] = {}
    with Path(ruta_movies).open(encoding="utf-8", newline="") as archivo:
        for registro in csv.DictReader(archivo):
            titulos_por_id[int(registro["movieId"])] = registro["title"]

    LOGGER.info("cargados %d títulos desde %s", len(titulos_por_id), ruta_movies)

    return titulos_por_id


def cargar_generos(ruta_movies: Path) -> dict[int, list[str]]:
    """Carga los géneros de cada película: id de película crudo → nombres.

    Salen de la columna `genres` de `movies.csv`, separados por `|`, tal
    como los escribe MovieLens (incluido `(no genres listed)`). Solo se usan
    para mostrar: el modelo no ve los géneros (las dimensiones latentes emergen del entrenamiento).
    """
    generos_por_id: dict[int, list[str]] = {}
    with Path(ruta_movies).open(encoding="utf-8", newline="") as archivo:
        for registro in csv.DictReader(archivo):
            generos_por_id[int(registro["movieId"])] = registro["genres"].split("|")

    LOGGER.info("cargados los géneros de %d películas desde %s", len(generos_por_id), ruta_movies)

    return generos_por_id


def construir_indice_a_titulo(
    id_pelicula_a_indice: dict[int, int], titulos_por_id: dict[int, str]
) -> dict[int, str]:
    """Invierte `id_pelicula_a_indice` y lo combina con `titulos_por_id`.

    Devuelve el mapeo índice de columna de V → título, para las películas que
    aparecen en `id_pelicula_a_indice` (típicamente, las que sobrevivieron al
    filtro por mínimo).
    """
    return {
        indice: titulos_por_id[id_pelicula]
        for id_pelicula, indice in id_pelicula_a_indice.items()
    }


@dataclass
class DatosPreparados:
    """Salida de `preparar_datos_movielens`: único punto de entrada de datos para `demo.py`.

    Atributos:
        calificaciones: `ConjuntoCalificaciones` ya filtrado por `umbral`.
        titulos_por_indice: índice de columna de V → título.
    """

    calificaciones: ConjuntoCalificaciones
    titulos_por_indice: dict[int, str]


def preparar_datos_movielens(
    ruta_ratings: Path, ruta_movies: Path, umbral: int, k: int
) -> DatosPreparados:
    """Orquesta carga, filtro y títulos en un único punto de entrada.

    Encadena `cargar_calificaciones`, `filtrar_por_minimo` (con `umbral`,
    validado contra `k`), `cargar_titulos` y `construir_indice_a_titulo`.
    """
    datos = cargar_calificaciones(ruta_ratings)
    datos_filtrados = filtrar_por_minimo(datos, umbral, k)
    titulos_por_id = cargar_titulos(ruta_movies)
    titulos_por_indice = construir_indice_a_titulo(
        datos_filtrados.id_pelicula_a_indice, titulos_por_id
    )

    return DatosPreparados(calificaciones=datos_filtrados, titulos_por_indice=titulos_por_indice)


def particionar(
    M: np.ndarray, fraccion_prueba: float, generador: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Parte Ω en entrenamiento y prueba.

    Se aplica sobre la M ya filtrada por umbral. Sortea con `generador`
    round(fraccion_prueba · |Ω|) pares de Ω, sin reposición, para prueba; el
    resto queda en entrenamiento. Después pasa a entrenamiento todo par de
    prueba cuyo usuario o película quedó sin calificaciones de
    entrenamiento, así que el tamaño real de prueba puede ser menor que la
    fracción pedida. R no se toca: devuelve las máscaras (M_ent, M_prueba),
    disjuntas y con unión M.
    """
    filas, columnas = np.nonzero(M)
    n_prueba = int(round(fraccion_prueba * len(filas)))
    elegidos = generador.choice(len(filas), size=n_prueba, replace=False)

    M_prueba = np.zeros_like(M, dtype=bool)
    M_prueba[filas[elegidos], columnas[elegidos]] = True
    M_ent = M & ~M_prueba

    # Un par de prueba cuyo usuario o película quedó sin
    # calificaciones de entrenamiento pasa a entrenamiento.
    filas_sin_ent = ~M_ent.any(axis=1)
    columnas_sin_ent = ~M_ent.any(axis=0)
    a_entrenamiento = M_prueba & (filas_sin_ent[:, None] | columnas_sin_ent[None, :])
    M_prueba = M_prueba & ~a_entrenamiento
    M_ent = M_ent | a_entrenamiento

    LOGGER.info(
        "partición (fracción de prueba pedida=%.2f): entrenamiento=%d, prueba=%d "
        "calificaciones (%d pares de prueba pasados a entrenamiento)",
        fraccion_prueba, int(M_ent.sum()), int(M_prueba.sum()), int(a_entrenamiento.sum()),
    )

    return M_ent, M_prueba
