"""Carga y preprocesamiento de MovieLens."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.errores import ErrorDatosInsuficientes

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
    """Carga un archivo estilo `u.data`.

    El archivo tiene una calificación por línea: usuario, película,
    calificación y timestamp separados por tab. Reindexa los ids de usuario
    y de película desde 0 (según su orden ascendente) y arma R (con NaN en
    los huecos) y M. No aplica ningún filtro.
    """
    filas: list[tuple[int, int, float]] = []
    with Path(ruta_archivo).open(encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:
                continue
            id_usuario_str, id_pelicula_str, calificacion_str, _timestamp = linea.split("\t")
            filas.append((int(id_usuario_str), int(id_pelicula_str), float(calificacion_str)))

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


def filtrar_por_minimo(datos: ConjuntoCalificaciones, k: int) -> ConjuntoCalificaciones:
    """Elimina filas y columnas con menos de k calificaciones.

    Repite la eliminación hasta que una pasada no elimine nada: sacar una
    columna puede dejar a una fila por debajo de k (o viceversa), así que
    una sola pasada no alcanza. Reconstruye `id_usuario_a_indice` y
    `id_pelicula_a_indice` desde cero para los ids que sobrevivieron, con
    índices nuevos y contiguos desde 0 (no reutiliza los índices viejos de
    `datos`, que quedan obsoletos apenas se elimina la primera fila o
    columna).

    Lanza `ErrorDatosInsuficientes` si el filtro deja la matriz sin filas o
    sin columnas.
    """
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

        filas_a_mantener = filas_activas[conteo_filas >= k]
        columnas_a_mantener = columnas_activas[conteo_columnas >= k]

        if len(filas_a_mantener) == len(filas_activas) and len(columnas_a_mantener) == len(
            columnas_activas
        ):
            break

        usuarios_eliminados += len(filas_activas) - len(filas_a_mantener)
        peliculas_eliminadas += len(columnas_activas) - len(columnas_a_mantener)
        LOGGER.debug(
            "filtro por k=%d, pasada %d: quedan %d usuarios y %d películas",
            k, n_pasada, len(filas_a_mantener), len(columnas_a_mantener),
        )

        if len(filas_a_mantener) == 0 or len(columnas_a_mantener) == 0:
            raise ErrorDatosInsuficientes(k)

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
            "filtro por k=%d: se eliminaron %d usuarios, %d películas y %d calificaciones",
            k, usuarios_eliminados, peliculas_eliminadas, calificaciones_eliminadas,
        )

    return ConjuntoCalificaciones(
        R=R_filtrado,
        M=M_filtrado,
        id_usuario_a_indice=nuevo_id_usuario_a_indice,
        id_pelicula_a_indice=nuevo_id_pelicula_a_indice,
    )
