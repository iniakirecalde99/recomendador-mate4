"""Verificaciones de la sección 8.1 y comparación de ALS con descenso de gradiente.

Comprueba los datos que cita la sección 8.1 (fechas, valores de calificación
presentes y el más frecuente) y compara los modelos de ALS y de descenso de
gradiente con k por defecto, entrenados con la misma división de
`experimentos/seccion8.py`: correlación entre sus V, entre sus predicciones
sobre los pares de prueba (Ω_P) y RMSE entre esas predicciones, y la
diagonal v_j1 + v_j2 de la sección 8.4. No modifica nada de `src/`.

Uso: `python -m experimentos.extra --help`
"""

import argparse
import csv
import datetime as dt
import logging
from collections import Counter

import numpy as np

from experimentos.seccion8 import construir_parser as construir_parser_seccion8
from experimentos.seccion8 import dividir_por_usuario, rmse
from src import config
from src.als import entrenar_als
from src.datos import cargar_calificaciones, filtrar_por_minimo
from src.gradiente import entrenar_gd
from src.modelo import inicializar_factores, predecir

LOGGER = logging.getLogger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Mismos argumentos que `experimentos.seccion8`; `--salida` y `--usuario` no se usan."""
    parser = construir_parser_seccion8()
    parser.prog = "python -m experimentos.extra"
    parser.description = "Verificaciones de 8.1 y comparación de ALS con descenso de gradiente."
    return parser


def correlacion(x: np.ndarray, y: np.ndarray) -> float:
    """Coeficiente de correlación de Pearson entre dos vectores."""
    return float(np.corrcoef(x, y)[0, 1])


def main(argv: list[str] | None = None) -> None:
    """Corre las verificaciones y las comparaciones, y las informa por logging."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("src").setLevel(logging.WARNING)
    args = construir_parser().parse_args(argv)

    # 8.1: fechas y valores de calificación
    marcas = []
    with (args.datos / "ratings.csv").open(encoding="utf-8", newline="") as archivo:
        for registro in csv.DictReader(archivo):
            marcas.append(int(registro["timestamp"]))
    LOGGER.info("fechas: %s a %s",
                dt.datetime.fromtimestamp(min(marcas), dt.timezone.utc),
                dt.datetime.fromtimestamp(max(marcas), dt.timezone.utc))

    crudo = cargar_calificaciones(args.datos / "ratings.csv")
    datos = filtrar_por_minimo(crudo, args.umbral, args.k)
    R, M = datos.R, datos.M
    m, n = R.shape
    M_ent, M_pru = dividir_por_usuario(M, args.proporcion_prueba, args.k, np.random.default_rng(args.semilla))
    R_ent = np.where(M_ent, R, np.nan)
    for nombre, matriz, mascara in (("sin filtrar", crudo.R, crudo.M), ("filtrado", R, M),
                                    ("entrenamiento", R, M_ent)):
        conteo = Counter(matriz[mascara].tolist())
        LOGGER.info("%s: %d valores distintos, más frecuentes %s", nombre, len(conteo), conteo.most_common(2))

    # ALS y descenso de gradiente desde las mismas U₀, V₀
    U0, V0 = inicializar_factores(m, n, args.k, config.ESCALA_INICIALIZACION_DEFECTO,
                                  np.random.default_rng(args.semilla))
    als = entrenar_als(R_ent, M_ent, U0.copy(), V0.copy(), args.epsilon, args.max_iter)
    gd = entrenar_gd(R_ent, M_ent, U0.copy(), V0.copy(), args.eta, args.epsilon, args.max_iter)

    for columna in range(args.k):
        LOGGER.info("correlación de la columna %d de V, ALS vs. GD: %.4f",
                    columna + 1, correlacion(als.V[:, columna], gd.V[:, columna]))

    R_hat_als = predecir(als.U, als.V)
    R_hat_gd = predecir(gd.U, gd.V)
    LOGGER.info("correlación de las predicciones ALS vs. GD sobre Ω_P: %.4f",
                correlacion(R_hat_als[M_pru], R_hat_gd[M_pru]))
    LOGGER.info("RMSE entre las predicciones de ALS y de GD sobre Ω_P: %.4f",
                rmse(R_hat_als, R_hat_gd, M_pru))
    LOGGER.info("correlación de las predicciones ALS vs. GD en todas las entradas: %.4f",
                correlacion(R_hat_als.ravel(), R_hat_gd.ravel()))

    # 8.4: la diagonal de V de ALS frente al promedio de cada película
    promedio_pelicula = np.nanmean(R, axis=0)
    LOGGER.info("correlación entre las dos columnas de V (ALS): %.4f", correlacion(als.V[:, 0], als.V[:, 1]))
    LOGGER.info("correlación de v_j1 + v_j2 con el promedio de cada película: %.4f",
                correlacion(als.V[:, 0] + als.V[:, 1], promedio_pelicula))


if __name__ == "__main__":
    main()
