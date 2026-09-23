"""Demo de la factorización R ≈ U·Vᵀ sobre MovieLens — único módulo con `print`."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src import config
from src.als import entrenar_als
from src.comparacion import comparar_metodos, formatear_tabla_comparacion, graficar_convergencia
from src.datos import preparar_datos_movielens
from src.errores import DivergenciaError, UsuarioNoEncontradoError
from src.gradiente import entrenar_gd
from src.modelo import inicializar_factores
from src.recomendaciones import extremos_por_factor, recomendar_top_n

LOGGER = logging.getLogger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Arma el parser de `python -m src.demo`.

    Defaults de `config.py`. Sin `--rmse` ni
    `--metodo`: la demo siempre corre ALS y GD, para poder compararlos.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.demo",
        description="Factorización R ≈ U·Vᵀ sobre MovieLens: ALS vs. descenso de gradiente.",
    )
    parser.add_argument("--datos", type=Path, default=config.RUTA_DATOS_DEFECTO)
    parser.add_argument("--grafico", type=Path, default=config.RUTA_GRAFICO_DEFECTO)
    parser.add_argument("--umbral", type=int, default=config.UMBRAL_DEFECTO)
    parser.add_argument("--k", type=int, default=config.K_DEFECTO)
    parser.add_argument("--eta", type=float, default=config.ETA_DEFECTO)
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_DEFECTO)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER_DEFECTO)
    parser.add_argument("--semilla", type=int, default=config.SEMILLA_DEFECTO)
    parser.add_argument("--usuario", type=int, default=config.USUARIO_DEFECTO)
    parser.add_argument("--top-n", type=int, default=config.TOP_N_DEFECTO)
    return parser


def _traducir_usuario(id_usuario_a_indice: dict[int, int], usuario_id: int) -> int:
    """Traduce un id crudo de MovieLens a índice de fila.

    Lanza `UsuarioNoEncontradoError` si `usuario_id` no está en el mapeo
    (no existe en MovieLens, o el filtro por `--k` lo eliminó).
    """
    indice = id_usuario_a_indice.get(usuario_id)
    if indice is None:
        raise UsuarioNoEncontradoError(usuario_id)
    return indice


def _formatear_top_n_lado_a_lado(
    recomendaciones_als: list[tuple[str, float]], recomendaciones_gd: list[tuple[str, float]]
) -> str:
    """Arma las recomendaciones de ALS y GD como dos columnas, lado a lado."""
    ancho = 40
    lineas = [f"{'ALS':<{ancho}}{'GD':<{ancho}}"]
    n_filas = max(len(recomendaciones_als), len(recomendaciones_gd))
    for i in range(n_filas):
        texto_als = ""
        if i < len(recomendaciones_als):
            titulo, r_hat = recomendaciones_als[i]
            texto_als = f"{titulo}: {r_hat:.2f}"
        texto_gd = ""
        if i < len(recomendaciones_gd):
            titulo, r_hat = recomendaciones_gd[i]
            texto_gd = f"{titulo}: {r_hat:.2f}"
        lineas.append(f"{texto_als:<{ancho}}{texto_gd:<{ancho}}")
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> None:
    """Orquesta la demo completa: datos, entrenamiento, comparación y recomendaciones."""
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO)

    args = construir_parser().parse_args(argv)

    datos = preparar_datos_movielens(
        args.datos / "u.data", args.datos / "u.item", args.umbral, args.k
    )
    R = datos.calificaciones.R
    M = datos.calificaciones.M
    m, n = R.shape
    print(
        f"MovieLens filtrado (umbral={args.umbral}, k={args.k}): {m} usuarios, "
        f"{n} películas, {int(M.sum())} calificaciones."
    )

    indice_usuario = _traducir_usuario(datos.calificaciones.id_usuario_a_indice, args.usuario)

    generador = np.random.default_rng(args.semilla)
    U0, V0 = inicializar_factores(
        m=m, n=n, k=args.k, escala=config.ESCALA_INICIALIZACION_DEFECTO, generador=generador
    )

    try:
        resultado_als = entrenar_als(
            R, M, U0.copy(), V0.copy(), epsilon=args.epsilon, max_iter=args.max_iter
        )
        resultado_gd = entrenar_gd(
            R, M, U0.copy(), V0.copy(), eta=args.eta, epsilon=args.epsilon, max_iter=args.max_iter
        )
    except DivergenciaError as error:
        print(f"\nError: {error}")
        return

    filas = comparar_metodos(resultado_als, resultado_gd)
    graficar_convergencia(resultado_als, resultado_gd, args.grafico)

    print()
    print(formatear_tabla_comparacion(filas))

    recomendaciones_als = recomendar_top_n(
        resultado_als.U,
        resultado_als.V,
        M,
        indice_usuario=indice_usuario,
        n=args.top_n,
        titulos_por_indice=datos.titulos_por_indice,
    )
    recomendaciones_gd = recomendar_top_n(
        resultado_gd.U,
        resultado_gd.V,
        M,
        indice_usuario=indice_usuario,
        n=args.top_n,
        titulos_por_indice=datos.titulos_por_indice,
    )
    print(f"\nTop-{args.top_n} recomendaciones para el usuario {args.usuario}:")
    print(_formatear_top_n_lado_a_lado(recomendaciones_als, recomendaciones_gd))

    print("\nExtremos por factor latente (según V de ALS; GD no se usa para esta parte):")
    for factor, mayores, menores in extremos_por_factor(resultado_als.V, datos.titulos_por_indice):
        print(f"  Factor {factor} — mayores:")
        for titulo, valor in mayores:
            print(f"    {titulo}: {valor:.3f}")
        print(f"  Factor {factor} — menores:")
        for titulo, valor in menores:
            print(f"    {titulo}: {valor:.3f}")


if __name__ == "__main__":
    main()
