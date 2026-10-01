"""Experimento 4: ordenar las recomendaciones por la estimación relativa.

Respalda el experimento 4 de experimentos/bitacora.md. Con el modelo de main (ALS,
k = 2, defaults de `config.py`, entrenado sobre todo Ω) compara dos
ordenamientos del top-N de cada usuario, entre las películas que no calificó:

- absoluto: r̂ᵢⱼ = Vⱼ · Uᵢ (el de main y del experimento 3);
- relativo: Vⱼ · (Uᵢ − ū), con ū el promedio de las filas de U entrenada, que
  es r̂ᵢⱼ menos el promedio de r̂ de la película j sobre todos los usuarios.

Mide la concentración y el promedio de r̂ absoluto de lo recomendado, con
usuarios reales (sobre los que se evalúa el criterio) y, como referencia,
con usuarios simulados con notas reales (el caso del front). No modifica
nada de `src/`.

Uso: `python -m experimentos.ordenamiento_relativo --help`
"""

import argparse
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experimentos import concentracion
from src import config
from src.als import entrenar_als, resolver_factor
from src.datos import preparar_datos_movielens
from src.modelo import inicializar_factores, predecir

LOGGER = logging.getLogger(__name__)

N_MAS_FRECUENTES_DEFECTO = 5
UMBRAL_FRECUENTE_DEFECTO = 0.2  # mismo umbral que los experimentos anteriores

# Criterio del experimento 4 (fijado antes de correr), sobre usuarios reales.
MAX_FRECUENCIA_CRITERIO = 0.25
MIN_DISTINTAS_CRITERIO = 127  # las del ordenamiento absoluto en el experimento 3
MAX_BAJA_PROMEDIO_R_HAT_CRITERIO = 0.5


@dataclass
class ResultadoOrdenamiento:
    """Concentración y nivel de r̂ del top-N con un ordenamiento.

    Atributos:
        n_usuarios, n_peliculas: tamaño del grupo y del catálogo.
        distintas_en_top_n: películas que aparecen en el top-N de algún usuario.
        mas_frecuente, frecuencia_mas_frecuente: la que aparece en más top-N y
            la fracción de usuarios en cuyo top-N aparece.
        n_frecuentes: películas en el top-N de más de `umbral_frecuente` de
            los usuarios.
        promedio_r_hat: promedio de r̂ᵢⱼ absoluto sobre todos los pares
            (usuario, película recomendada).
        mas_frecuentes: las más frecuentes, (título, fracción).
    """

    n_usuarios: int
    n_peliculas: int
    distintas_en_top_n: int
    mas_frecuente: str
    frecuencia_mas_frecuente: float
    n_frecuentes: int
    promedio_r_hat: float
    mas_frecuentes: list[tuple[str, float]]


def puntajes_relativos(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Puntaje relativo (U − ū)·Vᵀ, con ū el promedio de las filas de U.

    Es r̂ᵢⱼ menos el promedio de r̂ de la película j sobre todos los usuarios,
    porque promedio_i (Uᵢ · Vⱼ) = ū · Vⱼ.
    """
    u_barra = U.mean(axis=0)
    return predecir(U - u_barra, V)


def medir_tops(
    puntajes: np.ndarray,
    R_hat: np.ndarray,
    M: np.ndarray,
    titulos_por_indice: dict[int, str],
    top_n: int,
    umbral_frecuente: float,
    n_mas_frecuentes: int = N_MAS_FRECUENTES_DEFECTO,
) -> ResultadoOrdenamiento:
    """Top-N de cada fila según `puntajes`, sin lo calificado (M), y sus métricas.

    Ordena de mayor a menor puntaje con empates por índice (como
    `recomendaciones.recomendar_top_n`). El promedio de r̂ usa `R_hat`
    (estimación absoluta), no los puntajes.
    """
    puntajes = puntajes.copy()
    puntajes[M] = -np.inf  # lo ya calificado no se recomienda
    tops = np.argsort(-puntajes, axis=1, kind="stable")[:, :top_n]

    n_usuarios = puntajes.shape[0]
    apariciones: Counter[int] = Counter(tops.ravel().tolist())
    frecuencias = {j: c / n_usuarios for j, c in apariciones.items()}
    ordenadas = sorted(frecuencias.items(), key=lambda par: (-par[1], par[0]))
    j_max, f_max = ordenadas[0]
    r_hat_recomendadas = np.take_along_axis(R_hat, tops, axis=1)
    return ResultadoOrdenamiento(
        n_usuarios=n_usuarios,
        n_peliculas=puntajes.shape[1],
        distintas_en_top_n=len(apariciones),
        mas_frecuente=titulos_por_indice[j_max],
        frecuencia_mas_frecuente=f_max,
        n_frecuentes=sum(f > umbral_frecuente for f in frecuencias.values()),
        promedio_r_hat=float(r_hat_recomendadas.mean()),
        mas_frecuentes=[(titulos_por_indice[j], f) for j, f in ordenadas[:n_mas_frecuentes]],
    )


def simular_ambos_ordenamientos(
    U: np.ndarray,
    V: np.ndarray,
    M: np.ndarray,
    titulos_por_indice: dict[int, str],
    notas: np.ndarray,
    n_usuarios: int,
    min_calificadas: int,
    max_calificadas: int,
    n_a_calificar: int,
    top_n: int,
    semilla_simulacion: int,
    umbral_frecuente: float,
) -> tuple[ResultadoOrdenamiento, ResultadoOrdenamiento]:
    """Usuarios simulados como en `concentracion.simular` (sin centrado), con los dos ordenamientos.

    Repite el mismo sorteo que `concentracion.simular` (misma secuencia de
    llamadas al generador), resuelve u con V fija (informe 5) y ordena por
    V·u (absoluto) y por V·(u − ū) (relativo), con ū el promedio de las filas
    de U de los usuarios reales. Devuelve (absoluto, relativo).
    """
    n = V.shape[0]
    u_barra = U.mean(axis=0)
    a_calificar = np.argsort(-M.sum(axis=0), kind="stable")[:n_a_calificar]
    generador = np.random.default_rng(semilla_simulacion)

    R_hat = np.empty((n_usuarios, n))
    M_simulados = np.zeros((n_usuarios, n), dtype=bool)
    for s in range(n_usuarios):
        cantidad = generador.integers(min_calificadas, max_calificadas + 1)
        elegidas = generador.choice(a_calificar, size=cantidad, replace=False)
        r = np.full(n, np.nan)
        r[elegidas] = generador.choice(notas, size=cantidad)

        # Informe 5: paso de U de ALS con V fija, para el usuario nuevo.
        u = resolver_factor(r[None, :], ~np.isnan(r)[None, :], V)[0]
        R_hat[s] = V @ u
        M_simulados[s, elegidas] = True

    relativo = R_hat - V @ u_barra  # V·(u − ū)
    return (
        medir_tops(R_hat, R_hat, M_simulados, titulos_por_indice, top_n, umbral_frecuente),
        medir_tops(relativo, R_hat, M_simulados, titulos_por_indice, top_n, umbral_frecuente),
    )


def evaluar_criterio(
    absoluto: ResultadoOrdenamiento, relativo: ResultadoOrdenamiento
) -> dict[int, bool]:
    """Criterio del experimento 4 sobre usuarios reales (se cumple si se cumplen los tres).

    1. La más frecuente del relativo aparece en a lo sumo el 25 % de los top-N.
    2. Películas distintas del relativo: al menos 127.
    3. El promedio de r̂ absoluto de lo recomendado baja a lo sumo 0,5
       respecto del ordenamiento absoluto.
    """
    return {
        1: relativo.frecuencia_mas_frecuente <= MAX_FRECUENCIA_CRITERIO,
        2: relativo.distintas_en_top_n >= MIN_DISTINTAS_CRITERIO,
        3: absoluto.promedio_r_hat - relativo.promedio_r_hat <= MAX_BAJA_PROMEDIO_R_HAT_CRITERIO,
    }


def construir_parser() -> argparse.ArgumentParser:
    """Parámetros del experimento; los del modelo salen de `config.py` (modelo de main)."""
    parser = argparse.ArgumentParser(
        prog="python -m experimentos.ordenamiento_relativo",
        description="Ordenamiento absoluto vs. relativo del top-N con el modelo de main.",
    )
    parser.add_argument("--datos", type=Path, default=config.RUTA_DATOS_DEFECTO)
    parser.add_argument("--umbral", type=int, default=config.UMBRAL_DEFECTO)
    parser.add_argument("--k", type=int, default=config.K_DEFECTO)
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_DEFECTO)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER_DEFECTO)
    parser.add_argument("--semilla", type=int, default=config.SEMILLA_DEFECTO)
    parser.add_argument("--top-n", type=int, default=config.TOP_N_DEFECTO)
    parser.add_argument("--umbral-frecuente", type=float, default=UMBRAL_FRECUENTE_DEFECTO)
    return parser


def _reportar(nombre: str, resultado: ResultadoOrdenamiento, top_n: int) -> None:
    LOGGER.info(
        "  %s: %d de %d películas en algún top-%d | más frecuente: %s (%.1f%%) | "
        "en > 20%%: %d | promedio de r̂ recomendado: %.3f",
        nombre, resultado.distintas_en_top_n, resultado.n_peliculas, top_n,
        resultado.mas_frecuente, 100 * resultado.frecuencia_mas_frecuente,
        resultado.n_frecuentes, resultado.promedio_r_hat,
    )
    LOGGER.info(
        "    top-5: %s", "; ".join(f"{t} {100 * f:.1f}%" for t, f in resultado.mas_frecuentes)
    )


def main(argv: list[str] | None = None) -> None:
    """Entrena el modelo de main sobre todo Ω y compara los dos ordenamientos."""
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
    U, V = modelo.U, modelo.V
    LOGGER.info(
        "MovieLens filtrado (umbral=%d, k=%d): %d usuarios × %d películas; ALS: %d it (%s), "
        "SCE %.4f; ū = %s",
        args.umbral, args.k, m, n, modelo.n_iteraciones, modelo.motivo_corte,
        modelo.historial_f[-1], np.round(U.mean(axis=0), 4),
    )

    R_hat = predecir(U, V)
    titulos = datos.titulos_por_indice
    reales_abs = medir_tops(R_hat, R_hat, M, titulos, args.top_n, args.umbral_frecuente)
    reales_rel = medir_tops(
        puntajes_relativos(U, V), R_hat, M, titulos, args.top_n, args.umbral_frecuente
    )
    LOGGER.info("\nUsuarios reales (%d):", m)
    _reportar("absoluto", reales_abs, args.top_n)
    _reportar("relativo", reales_rel, args.top_n)

    criterios = evaluar_criterio(reales_abs, reales_rel)
    LOGGER.info(
        "Criterio (usuarios reales): %s; baja del promedio de r̂ = %.3f",
        ", ".join(f"{c}: {'sí' if ok else 'no'}" for c, ok in criterios.items()),
        reales_abs.promedio_r_hat - reales_rel.promedio_r_hat,
    )
    LOGGER.info("Resultado: %s", "SE CUMPLE" if all(criterios.values()) else "NO se cumple")

    simulados_abs, simulados_rel = simular_ambos_ordenamientos(
        U, V, M, titulos, concentracion.notas_reales(args.datos / "ratings.csv"),
        n_usuarios=concentracion.N_USUARIOS_DEFECTO,
        min_calificadas=concentracion.MIN_CALIFICADAS_DEFECTO,
        max_calificadas=concentracion.MAX_CALIFICADAS_DEFECTO,
        n_a_calificar=config.N_A_CALIFICAR_DEFECTO, top_n=args.top_n,
        semilla_simulacion=concentracion.SEMILLA_SIMULACION_DEFECTO,
        umbral_frecuente=args.umbral_frecuente,
    )
    LOGGER.info(
        "\nReferencia, fuera del criterio: usuarios simulados con notas reales (%d):",
        simulados_abs.n_usuarios,
    )
    _reportar("absoluto", simulados_abs, args.top_n)
    _reportar("relativo", simulados_rel, args.top_n)


if __name__ == "__main__":
    main()
