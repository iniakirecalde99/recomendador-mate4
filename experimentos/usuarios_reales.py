"""Experimento 3: concentración de las recomendaciones para los usuarios reales.

Respalda el experimento 3 de experimentos/bitacora.md. Con el modelo de main (ALS,
k = 2, defaults de `config.py`, entrenado sobre todo Ω), arma el top-N de
cada usuario real del conjunto filtrado con su fila de U entrenada, entre
las películas que no calificó, y mide cuánto se repiten las películas. Es la
contraparte, con usuarios reales, de `experimentos/concentracion.py` (que
usa usuarios simulados). No modifica nada de `src/`.

Uso: `python -m experimentos.usuarios_reales --help`
"""

import argparse
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config
from src.als import entrenar_als
from src.datos import preparar_datos_movielens
from src.modelo import inicializar_factores, predecir

LOGGER = logging.getLogger(__name__)

N_MAS_FRECUENTES_DEFECTO = 5
UMBRAL_FRECUENTE_DEFECTO = 0.2  # mismo umbral que experimentos/concentracion.py


@dataclass
class ResultadoUsuariosReales:
    """Concentración del top-N sobre los usuarios reales.

    Atributos:
        n_usuarios: cantidad de usuarios (filas de U).
        distintas_en_top_n: películas que aparecen en el top-N de algún usuario.
        n_peliculas: películas del conjunto filtrado.
        mas_frecuente, frecuencia_mas_frecuente: la película que aparece en
            más top-N y la fracción de usuarios en cuyo top-N aparece.
        mas_frecuentes: las `n_mas_frecuentes` más frecuentes, (título, fracción).
        frecuencias: índice de película → fracción de usuarios (todas las que
            aparecen alguna vez).
    """

    n_usuarios: int
    distintas_en_top_n: int
    n_peliculas: int
    mas_frecuente: str
    frecuencia_mas_frecuente: float
    mas_frecuentes: list[tuple[str, float]]
    frecuencias: dict[int, float]


def concentracion_usuarios_reales(
    U: np.ndarray,
    V: np.ndarray,
    M: np.ndarray,
    titulos_por_indice: dict[int, str],
    top_n: int,
    n_mas_frecuentes: int,
) -> ResultadoUsuariosReales:
    """Top-N de cada usuario real con su fila de U y cuenta cuánto se repiten las películas.

    Para cada usuario i, ordena las películas que no calificó (M[i, j] False)
    por r̂ᵢⱼ = (U·Vᵀ)ᵢⱼ de mayor a menor (empates por índice, como
    `recomendaciones.recomendar_top_n`) y toma las `top_n` primeras.
    """
    R_hat = predecir(U, V)
    R_hat[M] = -np.inf  # lo ya calificado no se recomienda
    tops = np.argsort(-R_hat, axis=1, kind="stable")[:, :top_n]

    n_usuarios = U.shape[0]
    apariciones: Counter[int] = Counter(tops.ravel().tolist())
    frecuencias = {j: c / n_usuarios for j, c in apariciones.items()}
    ordenadas = sorted(frecuencias.items(), key=lambda par: (-par[1], par[0]))
    j_max, f_max = ordenadas[0]
    return ResultadoUsuariosReales(
        n_usuarios=n_usuarios,
        distintas_en_top_n=len(apariciones),
        n_peliculas=V.shape[0],
        mas_frecuente=titulos_por_indice[j_max],
        frecuencia_mas_frecuente=f_max,
        mas_frecuentes=[(titulos_por_indice[j], f) for j, f in ordenadas[:n_mas_frecuentes]],
        frecuencias=frecuencias,
    )


def construir_parser() -> argparse.ArgumentParser:
    """Parámetros del experimento; los del modelo salen de `config.py` (modelo de main)."""
    parser = argparse.ArgumentParser(
        prog="python -m experimentos.usuarios_reales",
        description="Concentración del top-N de los usuarios reales con el modelo de main.",
    )
    parser.add_argument("--datos", type=Path, default=config.RUTA_DATOS_DEFECTO)
    parser.add_argument("--umbral", type=int, default=config.UMBRAL_DEFECTO)
    parser.add_argument("--k", type=int, default=config.K_DEFECTO)
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_DEFECTO)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER_DEFECTO)
    parser.add_argument("--semilla", type=int, default=config.SEMILLA_DEFECTO)
    parser.add_argument("--top-n", type=int, default=config.TOP_N_DEFECTO)
    parser.add_argument("--n-mas-frecuentes", type=int, default=N_MAS_FRECUENTES_DEFECTO)
    parser.add_argument("--umbral-frecuente", type=float, default=UMBRAL_FRECUENTE_DEFECTO)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entrena el modelo de main sobre todo Ω y reporta la concentración de los usuarios reales."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("src").setLevel(logging.WARNING)  # sin el progreso por iteración de ALS
    args = construir_parser().parse_args(argv)

    datos = preparar_datos_movielens(
        args.datos / "ratings.csv", args.datos / "movies.csv", args.umbral, args.k
    )
    R, M = datos.calificaciones.R, datos.calificaciones.M
    m, n = R.shape
    U0, V0 = inicializar_factores(
        m=m, n=n, k=args.k, escala=config.ESCALA_INICIALIZACION_DEFECTO,
        generador=np.random.default_rng(args.semilla),
    )
    modelo = entrenar_als(R, M, U0, V0, epsilon=args.epsilon, max_iter=args.max_iter)

    resultado = concentracion_usuarios_reales(
        modelo.U, modelo.V, M, datos.titulos_por_indice, args.top_n, args.n_mas_frecuentes
    )
    n_frecuentes = sum(f > args.umbral_frecuente for f in resultado.frecuencias.values())

    LOGGER.info(
        "MovieLens filtrado (umbral=%d, k=%d): %d usuarios × %d películas; ALS: %d it (%s), "
        "SCE %.4f",
        args.umbral, args.k, m, n, modelo.n_iteraciones, modelo.motivo_corte,
        modelo.historial_f[-1],
    )
    LOGGER.info(
        "Usuarios reales: %d de %d películas en algún top-%d | más frecuente: %s (%.1f%%) | "
        "en el top-%d de más del %.0f%% de los usuarios: %d",
        resultado.distintas_en_top_n, resultado.n_peliculas, args.top_n,
        resultado.mas_frecuente, 100 * resultado.frecuencia_mas_frecuente,
        args.top_n, 100 * args.umbral_frecuente, n_frecuentes,
    )
    LOGGER.info(
        "Top-%d más frecuentes: %s",
        args.n_mas_frecuentes,
        "; ".join(f"{t} {100 * f:.1f}%" for t, f in resultado.mas_frecuentes),
    )


if __name__ == "__main__":
    main()
