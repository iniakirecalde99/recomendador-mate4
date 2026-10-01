"""Experimento: concentración de las recomendaciones con k = 2, sin y con centrado.

Respalda la tabla de la sección 8 del informe (experimentos/bitacora.md, "Experimento
de concentración"). No es parte de la demo ni del modelo: el centrado por el
promedio global μ vive solo acá, para comparar; el camino principal (`src/`)
no centra.

Simula usuarios nuevos como los del recomendador HTML: cada uno califica entre
`--min-calificadas` y `--max-calificadas` de las películas más calificadas, se
resuelve su vector u con V de ALS fija (`resolver_factor`, informe 5) y se
cuenta qué películas entran en su top-N. Dos distribuciones de notas:
"uniforme" (enteros de 1 a 5 equiprobables) y "real" (notas de `ratings.csv`
redondeadas hacia arriba a enteros, como los botones de la página).

Uso: `python -m experimentos.concentracion --help`
"""

import argparse
import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config
from src.als import entrenar_als, resolver_factor
from src.datos import preparar_datos_movielens
from src.modelo import inicializar_factores, predecir

LOGGER = logging.getLogger(__name__)

N_USUARIOS_DEFECTO = 3000
MIN_CALIFICADAS_DEFECTO = 5
MAX_CALIFICADAS_DEFECTO = 10
SEMILLA_SIMULACION_DEFECTO = 0
UMBRAL_FRECUENTE_DEFECTO = 0.2  # "película frecuente": en el top-N de más del 20% de los usuarios
NOTA_MINIMA = 1
NOTA_MAXIMA = 5


@dataclass
class ResultadoSimulacion:
    """Resumen de una simulación (una fila de la tabla de la bitácora)."""

    mu: float
    sce_als: float
    iteraciones_als: int
    max_r_hat_fuera_omega: float
    distintas_en_top_n: int
    n_peliculas: int
    mas_frecuente: str
    frecuencia_mas_frecuente: float
    n_frecuentes: int
    angulo_u_p5_p50_p95: tuple[float, float, float]
    top_5: list[tuple[str, float]]


def notas_reales(ruta_ratings: Path) -> np.ndarray:
    """Notas de `ratings.csv` redondeadas hacia arriba a enteros de 1 a 5 (como los botones)."""
    with ruta_ratings.open(encoding="utf-8", newline="") as archivo:
        notas = np.array([float(fila["rating"]) for fila in csv.DictReader(archivo)])
    return np.clip(np.ceil(notas), NOTA_MINIMA, NOTA_MAXIMA)


def simular(
    R: np.ndarray,
    M: np.ndarray,
    titulos_por_indice: dict[int, str],
    centrar: bool,
    notas: np.ndarray | None,
    k: int,
    epsilon: float,
    max_iter: int,
    semilla: int,
    n_usuarios: int,
    min_calificadas: int,
    max_calificadas: int,
    n_a_calificar: int,
    top_n: int,
    semilla_simulacion: int,
    umbral_frecuente: float,
) -> ResultadoSimulacion:
    """Entrena ALS (sobre R o sobre R − μ) y simula `n_usuarios` usuarios nuevos.

    Con `centrar`, μ es el promedio de las calificaciones observadas (Ω):
    se entrena sobre R − μ, el usuario nuevo se resuelve con sus notas − μ y
    la predicción es μ + Vⱼ·u. `notas=None` sortea notas uniformes de 1 a 5;
    si no, sortea de ese arreglo.
    """
    mu = float(np.mean(R[M])) if centrar else 0.0
    m, n = R.shape
    U0, V0 = inicializar_factores(
        m=m, n=n, k=k, escala=config.ESCALA_INICIALIZACION_DEFECTO,
        generador=np.random.default_rng(semilla),
    )
    resultado_als = entrenar_als(R - mu, M, U0, V0, epsilon=epsilon, max_iter=max_iter)
    V = resultado_als.V
    max_fuera = float(np.abs(mu + predecir(resultado_als.U, V))[~M].max())

    a_calificar = np.argsort(-M.sum(axis=0), kind="stable")[:n_a_calificar]
    generador = np.random.default_rng(semilla_simulacion)
    apariciones: Counter[int] = Counter()
    angulos = []
    for _ in range(n_usuarios):
        cantidad = generador.integers(min_calificadas, max_calificadas + 1)
        elegidas = generador.choice(a_calificar, size=cantidad, replace=False)
        r = np.full(n, np.nan)
        if notas is None:
            r[elegidas] = generador.integers(NOTA_MINIMA, NOTA_MAXIMA + 1, size=cantidad)
        else:
            r[elegidas] = generador.choice(notas, size=cantidad)

        # Informe 5: paso de U de ALS con V fija, para el usuario nuevo.
        u = resolver_factor((r - mu)[None, :], ~np.isnan(r)[None, :], V)[0]
        if k == 2:
            angulos.append(np.degrees(np.arctan2(u[1], u[0])))
        r_hat = mu + V @ u
        r_hat[elegidas] = -np.inf
        apariciones.update(np.argsort(-r_hat)[:top_n].tolist())

    frecuencias = {j: c / n_usuarios for j, c in apariciones.items()}
    j_max = max(frecuencias, key=frecuencias.get)
    percentiles = tuple(float(x) for x in np.percentile(angulos, [5, 50, 95])) if angulos else (np.nan,) * 3
    return ResultadoSimulacion(
        mu=mu,
        sce_als=resultado_als.historial_f[-1],
        iteraciones_als=resultado_als.n_iteraciones,
        max_r_hat_fuera_omega=max_fuera,
        distintas_en_top_n=len(apariciones),
        n_peliculas=n,
        mas_frecuente=titulos_por_indice[j_max],
        frecuencia_mas_frecuente=frecuencias[j_max],
        n_frecuentes=sum(f > umbral_frecuente for f in frecuencias.values()),
        angulo_u_p5_p50_p95=percentiles,
        top_5=[(titulos_por_indice[j], c / n_usuarios) for j, c in apariciones.most_common(5)],
    )


def construir_parser() -> argparse.ArgumentParser:
    """Parámetros del experimento; los del modelo salen de `config.py`."""
    parser = argparse.ArgumentParser(
        prog="python -m experimentos.concentracion",
        description="Concentración de las recomendaciones con k = 2, sin y con centrado por μ global.",
    )
    parser.add_argument("--datos", type=Path, default=config.RUTA_DATOS_DEFECTO)
    parser.add_argument("--umbral", type=int, default=config.UMBRAL_DEFECTO)
    parser.add_argument("--k", type=int, default=config.K_DEFECTO)
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_DEFECTO)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER_DEFECTO)
    parser.add_argument("--semilla", type=int, default=config.SEMILLA_DEFECTO)
    parser.add_argument("--n-usuarios", type=int, default=N_USUARIOS_DEFECTO)
    parser.add_argument("--min-calificadas", type=int, default=MIN_CALIFICADAS_DEFECTO)
    parser.add_argument("--max-calificadas", type=int, default=MAX_CALIFICADAS_DEFECTO)
    parser.add_argument("--n-a-calificar", type=int, default=config.N_A_CALIFICAR_DEFECTO)
    parser.add_argument("--top-n", type=int, default=config.TOP_N_DEFECTO)
    parser.add_argument("--semilla-simulacion", type=int, default=SEMILLA_SIMULACION_DEFECTO)
    parser.add_argument("--umbral-frecuente", type=float, default=UMBRAL_FRECUENTE_DEFECTO)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Corre las cuatro simulaciones (notas uniformes/reales × sin/con centrado) y las reporta."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("src").setLevel(logging.WARNING)  # sin el progreso por iteración de ALS
    args = construir_parser().parse_args(argv)

    datos = preparar_datos_movielens(
        args.datos / "ratings.csv", args.datos / "movies.csv", args.umbral, args.k
    )
    R, M = datos.calificaciones.R, datos.calificaciones.M
    distribuciones = {"uniforme": None, "real": notas_reales(args.datos / "ratings.csv")}

    LOGGER.info(
        "MovieLens filtrado (umbral=%d, k=%d): %d usuarios × %d películas; %d usuarios simulados",
        args.umbral, args.k, R.shape[0], R.shape[1], args.n_usuarios,
    )
    for nombre, notas in distribuciones.items():
        for centrar in (False, True):
            res = simular(
                R, M, datos.titulos_por_indice, centrar, notas,
                k=args.k, epsilon=args.epsilon, max_iter=args.max_iter, semilla=args.semilla,
                n_usuarios=args.n_usuarios, min_calificadas=args.min_calificadas,
                max_calificadas=args.max_calificadas, n_a_calificar=args.n_a_calificar,
                top_n=args.top_n, semilla_simulacion=args.semilla_simulacion,
                umbral_frecuente=args.umbral_frecuente,
            )
            p5, p50, p95 = res.angulo_u_p5_p50_p95
            LOGGER.info(
                "\nnotas %s, %s (μ=%.3f): SCE ALS=%.0f (%d it), max|r̂| fuera de Ω=%.2f",
                nombre, "CON centrado" if centrar else "SIN centrado", res.mu,
                res.sce_als, res.iteraciones_als, res.max_r_hat_fuera_omega,
            )
            LOGGER.info(
                "  distintas en algún top-%d: %d de %d | más frecuente: %s (%.1f%%) | "
                "en el top-%d de más del %.0f%% de los usuarios: %d | ángulo de u p5/p50/p95: %.0f°/%.0f°/%.0f°",
                args.top_n, res.distintas_en_top_n, res.n_peliculas, res.mas_frecuente,
                100 * res.frecuencia_mas_frecuente, args.top_n, 100 * args.umbral_frecuente,
                res.n_frecuentes, p5, p50, p95,
            )
            LOGGER.info("  top-5: %s", "; ".join(f"{t} {100 * f:.0f}%" for t, f in res.top_5))


if __name__ == "__main__":
    main()
