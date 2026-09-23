"""Demo de la factorización R ≈ U·Vᵀ sobre MovieLens — único módulo con `print`."""

import argparse
import logging
from pathlib import Path

import numpy as np

from src import config
from src.als import entrenar_als
from src.comparacion import (
    FilaComparacion,
    comparar_metodos,
    formatear_tabla_comparacion,
    graficar_convergencia,
)
from src.datos import preparar_datos_movielens
from src.gradiente import entrenar_gd
from src.modelo import ResultadoEntrenamiento, inicializar_factores
from src.recomendaciones import extremos_por_factor, recomendar_top_n

RUTA_DATOS_DEFECTO = Path("data/ml-100k")
RUTA_GRAFICO_DEFECTO = Path("convergencia.png")


def construir_parser() -> argparse.ArgumentParser:
    """Arma el parser de `python -m src.demo`.

    Defaults de `config.py`. Sin `--rmse`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.demo",
        description="Factorización R ≈ U·Vᵀ sobre MovieLens: ALS vs. descenso de gradiente.",
    )
    parser.add_argument("--metodo", choices=["als", "gd", "ambos"], default="ambos")
    parser.add_argument("--k", type=int, default=config.K_DEFECTO)
    parser.add_argument("--eta", type=float, default=config.ETA_DEFECTO)
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_DEFECTO)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER_DEFECTO)
    parser.add_argument("--semilla", type=int, default=config.SEMILLA_DEFECTO)
    parser.add_argument("--usuario", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=config.TOP_N_DEFECTO)
    return parser


def _fila_individual(metodo: str, resultado: ResultadoEntrenamiento) -> FilaComparacion:
    """Arma una `FilaComparacion` para un solo método (cuando `--metodo` no es "ambos")."""
    return FilaComparacion(
        metodo=metodo,
        n_iteraciones=resultado.n_iteraciones,
        motivo_corte=resultado.motivo_corte,
        tiempo_segundos=resultado.tiempo_segundos,
        sce_final=resultado.historial_f[-1],
    )


def main(argv: list[str] | None = None) -> None:
    """Orquesta la demo completa: datos, entrenamiento, comparación y recomendaciones."""
    logging.basicConfig(level=logging.INFO)

    args = construir_parser().parse_args(argv)

    datos = preparar_datos_movielens(
        RUTA_DATOS_DEFECTO / "u.data", RUTA_DATOS_DEFECTO / "u.item", args.k
    )
    R = datos.calificaciones.R
    M = datos.calificaciones.M
    m, n = R.shape
    print(
        f"MovieLens filtrado (k={args.k}): {m} usuarios, {n} películas, "
        f"{int(M.sum())} calificaciones."
    )

    generador = np.random.default_rng(args.semilla)
    U0, V0 = inicializar_factores(
        m=m, n=n, k=args.k, escala=config.ESCALA_INICIALIZACION_DEFECTO, generador=generador
    )

    resultado_als: ResultadoEntrenamiento | None = None
    resultado_gd: ResultadoEntrenamiento | None = None
    if args.metodo in ("als", "ambos"):
        resultado_als = entrenar_als(
            R, M, U0.copy(), V0.copy(), epsilon=args.epsilon, max_iter=args.max_iter
        )
    if args.metodo in ("gd", "ambos"):
        resultado_gd = entrenar_gd(
            R, M, U0.copy(), V0.copy(), eta=args.eta, epsilon=args.epsilon, max_iter=args.max_iter
        )

    if args.metodo == "ambos":
        filas = comparar_metodos(resultado_als, resultado_gd)
        graficar_convergencia(resultado_als, resultado_gd, RUTA_GRAFICO_DEFECTO)
    elif args.metodo == "als":
        filas = [_fila_individual("ALS", resultado_als)]
    else:
        filas = [_fila_individual("GD", resultado_gd)]

    print()
    print(formatear_tabla_comparacion(filas))

    # Para las recomendaciones, se usa ALS si corrió; si no, GD.
    resultado_para_recomendar = resultado_als if resultado_als is not None else resultado_gd

    recomendaciones = recomendar_top_n(
        resultado_para_recomendar.U,
        resultado_para_recomendar.V,
        M,
        indice_usuario=args.usuario,
        n=args.top_n,
        titulos_por_indice=datos.titulos_por_indice,
    )
    print(f"\nTop-{args.top_n} recomendaciones para el usuario {args.usuario}:")
    for titulo, r_hat in recomendaciones:
        print(f"  {titulo}: {r_hat:.2f}")

    print("\nExtremos por factor latente:")
    for factor, mayores, menores in extremos_por_factor(
        resultado_para_recomendar.V, datos.titulos_por_indice
    ):
        print(f"  Factor {factor} — mayores:")
        for titulo, valor in mayores:
            print(f"    {titulo}: {valor:.3f}")
        print(f"  Factor {factor} — menores:")
        for titulo, valor in menores:
            print(f"    {titulo}: {valor:.3f}")


if __name__ == "__main__":
    main()
