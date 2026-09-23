"""Carga y preprocesamiento de MovieLens."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
