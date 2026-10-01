"""Experimento de regularización: barrido de (k, λ) y veredicto.

Vive fuera de `src/`: usa `entrenar_als`, `entrenar_gd`, `resolver_factor`,
`f_regularizada` y `particionar` de `src/` (que con λ = 0 son main) y agrega
lo propio del experimento: métricas, criterios de éxito, barrido y
selección y la verificación con descenso de gradiente.

Uso: `python -m experimentos.regularizacion --help`
"""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config
from src.als import entrenar_als
from src.datos import particionar, preparar_datos_movielens
from src.errores import SistemaSingularError
from src.modelo import inicializar_factores, predecir, sce

LOGGER = logging.getLogger(__name__)


# --- Métricas ---


def medir_fuera_de_rango(
    U: np.ndarray, V: np.ndarray, M: np.ndarray, escala_min: float, escala_max: float
) -> tuple[float, float]:
    """Fracción de estimaciones fuera de rango y máximo de |r̂|.

    Mira solo los pares no observados (M False, con M la máscara de todo Ω:
    ni entrenamiento ni prueba). Una estimación está fuera de rango si cae
    fuera de [escala_min − 1, escala_max + 1]; los bordes cuentan como
    dentro. Devuelve (fracción entre 0 y 1, max |r̂| sobre esos pares).
    """
    r_hat_no_observados = predecir(U, V)[~M]
    fuera = (r_hat_no_observados < escala_min - 1) | (r_hat_no_observados > escala_max + 1)
    return float(fuera.mean()), float(np.abs(r_hat_no_observados).max())


# --- Criterios de éxito ---


@dataclass
class Veredicto:
    """Resultado de evaluar los cuatro criterios.

    Atributos:
        criterios: número de criterio (1 a 4) → si se cumple.
    """

    criterios: dict[int, bool]

    @property
    def exitoso(self) -> bool:
        """El experimento es exitoso si se cumplen los cuatro criterios."""
        return all(self.criterios.values())


def evaluar_criterios(
    k: int,
    sce_prueba_elegido: float,
    sce_prueba_k2_lambda0: float,
    fraccion_fuera_de_rango: float,
    peliculas_en_algun_top: int,
    frecuencia_mas_frecuente: float,
    min_peliculas: int = config.MIN_PELICULAS_EXITO,
    max_frecuencia: float = config.MAX_FRECUENCIA_EXITO,
    max_fraccion_fuera: float = config.MAX_FRACCION_FUERA_DE_RANGO,
) -> Veredicto:
    """Evalúa el par elegido contra los criterios de éxito.

    El criterio 2 usa SCE de prueba (de la partición); los criterios 3 y 4
    usan el modelo reentrenado sobre todo Ω.
    1. k > 2.
    2. SCE de prueba del par elegido < la de (k = 2, λ = 0).
    3. Fracción fuera de rango <= max_fraccion_fuera.
    4. Películas en algún top-10 >= min_peliculas y la más frecuente en a lo
       sumo max_frecuencia de los usuarios.
    """
    return Veredicto(
        criterios={
            1: k > 2,
            2: sce_prueba_elegido < sce_prueba_k2_lambda0,
            3: fraccion_fuera_de_rango <= max_fraccion_fuera,
            4: peliculas_en_algun_top >= min_peliculas
            and frecuencia_mas_frecuente <= max_frecuencia,
        }
    )


# --- Barrido y selección ---


@dataclass
class FilaBarrido:
    """Una fila de la tabla del barrido: un par (k, λ) entrenado con ALS sobre Ω_ent.

    Atributos:
        k, lambda_: el par.
        iteraciones, motivo_corte: del entrenamiento.
        f_final: último valor de `f_regularizada` sobre Ω_ent.
        sce_prueba: SCE pura sobre Ω_prueba.
        fraccion_fuera_de_rango, max_abs_fuera_de_omega: sobre los
            pares no observados.
        error: mensaje si el entrenamiento falló (por ejemplo, sistema
            singular); None si terminó. Las filas con error no entran en la
            selección.
    """

    k: int
    lambda_: float
    iteraciones: int
    motivo_corte: str
    f_final: float
    sce_prueba: float
    fraccion_fuera_de_rango: float
    max_abs_fuera_de_omega: float
    error: str | None = None


def barrer(
    R: np.ndarray,
    M: np.ndarray,
    M_ent: np.ndarray,
    M_prueba: np.ndarray,
    grilla_k: tuple[int, ...],
    grilla_lambda: tuple[float, ...],
    semilla_inicializacion: int,
    escala_inicializacion: float,
    epsilon: float,
    max_iter: int,
    escala_min: float,
    escala_max: float,
) -> list[FilaBarrido]:
    """Entrena ALS sobre Ω_ent para cada par (k, λ) de la grilla.

    Para cada k, U₀ y V₀ salen de un generador nuevo con
    `semilla_inicializacion`, así que todos los λ de un mismo k arrancan del
    mismo U₀, V₀. Un `SistemaSingularError` queda registrado en la fila y no
    corta el barrido.
    """
    m, n = R.shape
    filas: list[FilaBarrido] = []
    for k in grilla_k:
        U0, V0 = inicializar_factores(
            m=m, n=n, k=k, escala=escala_inicializacion,
            generador=np.random.default_rng(semilla_inicializacion),
        )
        for lambda_ in grilla_lambda:
            try:
                resultado = entrenar_als(
                    R, M_ent, U0.copy(), V0.copy(),
                    epsilon=epsilon, max_iter=max_iter, lambda_=lambda_,
                )
            except SistemaSingularError as error:
                LOGGER.warning("barrido: (k=%d, λ=%g) falló: %s", k, lambda_, error)
                filas.append(FilaBarrido(
                    k=k, lambda_=lambda_, iteraciones=0, motivo_corte="error",
                    f_final=math.nan, sce_prueba=math.nan,
                    fraccion_fuera_de_rango=math.nan, max_abs_fuera_de_omega=math.nan,
                    error=str(error),
                ))
                continue
            fraccion, maximo = medir_fuera_de_rango(
                resultado.U, resultado.V, M, escala_min, escala_max
            )
            filas.append(FilaBarrido(
                k=k, lambda_=lambda_, iteraciones=resultado.n_iteraciones,
                motivo_corte=resultado.motivo_corte, f_final=resultado.historial_f[-1],
                sce_prueba=sce(R, M_prueba, resultado.U, resultado.V),
                fraccion_fuera_de_rango=fraccion, max_abs_fuera_de_omega=maximo,
            ))
            LOGGER.info(
                "barrido: (k=%d, λ=%g) %d it, SCE de prueba=%.4f, fuera de rango=%.2f%%",
                k, lambda_, resultado.n_iteraciones, filas[-1].sce_prueba, 100 * fraccion,
            )
    return filas


def elegir_par(filas: list[FilaBarrido], tolerancia_empate: float) -> FilaBarrido:
    """Elige el par con menor SCE de prueba, con la regla de empate.

    Entre los pares sin error cuya SCE de prueba está a menos de
    `tolerancia_empate` (relativa) de la mejor, gana el de menor k; si hay
    varios con ese k, el de menor SCE de prueba.
    """
    validas = [f for f in filas if f.error is None]
    mejor = min(f.sce_prueba for f in validas)
    empatadas = [f for f in validas if f.sce_prueba < mejor * (1 + tolerancia_empate)]
    return min(empatadas, key=lambda f: (f.k, f.sce_prueba))


# --- CLI ---


def construir_parser() -> argparse.ArgumentParser:
    """Parámetros del experimento; todos con defaults de `config.py`."""
    parser = argparse.ArgumentParser(
        prog="python -m experimentos.regularizacion",
        description="Barrido de (k, λ) con ALS regularizado sobre MovieLens.",
    )
    parser.add_argument("--datos", type=Path, default=config.RUTA_DATOS_DEFECTO)
    parser.add_argument("--umbral", type=int, default=config.UMBRAL_DEFECTO)
    parser.add_argument("--grilla-k", type=int, nargs="+", default=list(config.GRILLA_K))
    parser.add_argument(
        "--grilla-lambda", type=float, nargs="+", default=list(config.GRILLA_LAMBDA)
    )
    parser.add_argument("--fraccion-prueba", type=float, default=config.FRACCION_PRUEBA)
    parser.add_argument("--semilla-particion", type=int, default=config.SEMILLA_PARTICION)
    parser.add_argument(
        "--semilla-inicializacion", type=int, default=config.SEMILLA_INICIALIZACION
    )
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_DEFECTO)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER_DEFECTO)
    parser.add_argument("--escala-min", type=float, default=config.ESCALA_MIN)
    parser.add_argument("--escala-max", type=float, default=config.ESCALA_MAX)
    parser.add_argument("--tolerancia-empate", type=float, default=config.TOLERANCIA_EMPATE)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Carga MovieLens, parte Ω, barre (k, λ) con ALS y reporta la tabla y el par elegido."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("src.als").setLevel(logging.WARNING)  # sin progreso por iteración
    args = construir_parser().parse_args(argv)

    # El filtro valida umbral >= k; se valida contra el mayor k de la grilla.
    datos = preparar_datos_movielens(
        args.datos / "ratings.csv", args.datos / "movies.csv", args.umbral, max(args.grilla_k)
    )
    R, M = datos.calificaciones.R, datos.calificaciones.M
    M_ent, M_prueba = particionar(
        M, args.fraccion_prueba, np.random.default_rng(args.semilla_particion)
    )

    filas = barrer(
        R, M, M_ent, M_prueba,
        grilla_k=tuple(args.grilla_k), grilla_lambda=tuple(args.grilla_lambda),
        semilla_inicializacion=args.semilla_inicializacion,
        escala_inicializacion=config.ESCALA_INICIALIZACION_DEFECTO,
        epsilon=args.epsilon, max_iter=args.max_iter,
        escala_min=args.escala_min, escala_max=args.escala_max,
    )
    elegido = elegir_par(filas, args.tolerancia_empate)

    LOGGER.info(
        "\n| k | λ | iteraciones | corte | f final | SCE de prueba | fuera de rango "
        "| max|r̂| fuera de Ω | error |"
    )
    LOGGER.info("|---:|---:|---:|---|---:|---:|---:|---:|---|")
    for f in filas:
        LOGGER.info(
            "| %d | %g | %d | %s | %.4f | %.4f | %.2f %% | %.2f | %s |",
            f.k, f.lambda_, f.iteraciones, f.motivo_corte, f.f_final, f.sce_prueba,
            100 * f.fraccion_fuera_de_rango, f.max_abs_fuera_de_omega, f.error or "",
        )
    LOGGER.info(
        "\nPar elegido: k=%d, λ=%g (SCE de prueba %.4f; optimista: la selección se hizo "
        "sobre el mismo conjunto de prueba)",
        elegido.k, elegido.lambda_, elegido.sce_prueba,
    )


if __name__ == "__main__":
    main()
