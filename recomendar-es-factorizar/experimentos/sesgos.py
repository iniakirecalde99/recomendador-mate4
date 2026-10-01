"""Experimento de sesgos: barrido de (k, λ), precisión@10 y concentración.

Vive fuera de `src/`: usa el modelo con sesgos de `src/sesgos.py` y, como
línea de base, `entrenar_als` de main. Mide SCE de prueba, fuera de rango,
precisión@10 y concentración con los dos ordenamientos, y
evalúa el criterio. Esta rama no usa descenso de gradiente.

Uso: `python -m experimentos.sesgos --help`
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config
from src.als import entrenar_als
from src.datos import particionar, preparar_datos_movielens
from src.modelo import inicializar_factores, predecir
from src.sesgos import (
    calcular_mu,
    entrenar_als_sesgos,
    predecir_con_sesgos,
    puntajes_ordenamiento_a,
    puntajes_ordenamiento_b,
)

LOGGER = logging.getLogger(__name__)


# --- Métricas ---


def precision_en_n(
    puntajes: np.ndarray,
    M_ent: np.ndarray,
    R: np.ndarray,
    M_prueba: np.ndarray,
    top_n: int,
    umbral_relevante: float,
) -> float:
    """Precisión@N promedio sobre los usuarios con algo relevante en prueba.

    Para cada usuario con al menos una calificación >= `umbral_relevante` en
    Ω_prueba, toma el top-N por `puntajes` entre las películas que no tiene
    en Ω_ent (empates por índice) y calcula aciertos / N, donde un acierto es
    una película del top-N calificada >= `umbral_relevante` en Ω_prueba.
    Devuelve el promedio sobre esos usuarios.
    """
    relevantes = M_prueba & (np.where(M_prueba, R, -np.inf) >= umbral_relevante)
    usuarios = np.where(relevantes.any(axis=1))[0]

    puntajes = puntajes.copy()
    puntajes[M_ent] = -np.inf  # lo de entrenamiento no se recomienda
    tops = np.argsort(-puntajes, axis=1, kind="stable")[:, :top_n]

    aciertos = np.take_along_axis(relevantes, tops, axis=1)[usuarios].sum(axis=1)
    return float(np.mean(aciertos / top_n))


def sce_sobre(R: np.ndarray, M: np.ndarray, R_hat: np.ndarray) -> float:
    """SCE de una estimación R̂ sobre la máscara M (SCE sobre Ω_prueba)."""
    error = np.where(M, R - R_hat, 0.0)
    return float(np.sum(error**2))


def medir_fuera_de_rango_r_hat(
    R_hat: np.ndarray, M: np.ndarray, escala_min: float, escala_max: float
) -> tuple[float, float]:
    """Fracción fuera de [escala_min − 1, escala_max + 1] y max |r̂|.

    Sobre los pares no observados (M False, con M la máscara de todo Ω) y
    siempre sobre r̂ completo (μ + bᵢ + cⱼ + Uᵢ·Vⱼ). Los bordes cuentan como
    dentro.
    """
    r_hat_no_observados = R_hat[~M]
    fuera = (r_hat_no_observados < escala_min - 1) | (r_hat_no_observados > escala_max + 1)
    return float(fuera.mean()), float(np.abs(r_hat_no_observados).max())


# --- Barrido, selección y línea de base ---


@dataclass
class FilaSesgos:
    """Una fila del barrido: un par (k, λ) entrenado con sesgos sobre Ω_ent.

    Atributos:
        k, lambda_: el par.
        iteraciones, motivo_corte, f_final: del entrenamiento (f = `f_sesgos`).
        sce_prueba: SCE sobre Ω_prueba de r̂ completo.
        fraccion_fuera_de_rango, max_abs_fuera_de_omega: sobre los pares no
            observados, con r̂ completo.
        precision_a, precision_b: precisión@N con los ordenamientos A y B.
    """

    k: int
    lambda_: float
    iteraciones: int
    motivo_corte: str
    f_final: float
    sce_prueba: float
    fraccion_fuera_de_rango: float
    max_abs_fuera_de_omega: float
    precision_a: float
    precision_b: float


def barrer_sesgos(
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
    top_n: int,
    umbral_relevante: float,
) -> list[FilaSesgos]:
    """Entrena ALS con sesgos sobre Ω_ent para cada par (k, λ).

    μ es el promedio de Ω_ent. Para cada k, U₀ y V₀ salen de un generador
    nuevo con `semilla_inicializacion` (los sesgos arrancan en 0).
    """
    m, n = R.shape
    mu = calcular_mu(R, M_ent)
    filas: list[FilaSesgos] = []
    for k in grilla_k:
        U0, V0 = inicializar_factores(
            m=m, n=n, k=k, escala=escala_inicializacion,
            generador=np.random.default_rng(semilla_inicializacion),
        )
        for lambda_ in grilla_lambda:
            modelo = entrenar_als_sesgos(
                R, M_ent, U0.copy(), V0.copy(),
                mu=mu, epsilon=epsilon, max_iter=max_iter, lambda_=lambda_,
            )
            R_hat = predecir_con_sesgos(modelo.U, modelo.V, modelo.b, modelo.c, mu)
            fraccion, maximo = medir_fuera_de_rango_r_hat(R_hat, M, escala_min, escala_max)
            filas.append(FilaSesgos(
                k=k, lambda_=lambda_, iteraciones=modelo.n_iteraciones,
                motivo_corte=modelo.motivo_corte, f_final=modelo.historial_f[-1],
                sce_prueba=sce_sobre(R, M_prueba, R_hat),
                fraccion_fuera_de_rango=fraccion, max_abs_fuera_de_omega=maximo,
                precision_a=precision_en_n(
                    puntajes_ordenamiento_a(modelo.U, modelo.V, modelo.b, modelo.c, mu),
                    M_ent, R, M_prueba, top_n, umbral_relevante,
                ),
                precision_b=precision_en_n(
                    puntajes_ordenamiento_b(modelo.U, modelo.V),
                    M_ent, R, M_prueba, top_n, umbral_relevante,
                ),
            ))
            LOGGER.info(
                "barrido: (k=%d, λ=%g) %d it, SCE de prueba=%.4f, precisión A=%.4f B=%.4f",
                k, lambda_, modelo.n_iteraciones, filas[-1].sce_prueba,
                filas[-1].precision_a, filas[-1].precision_b,
            )
    return filas


def elegir_par_sesgos(filas: list[FilaSesgos]) -> FilaSesgos:
    """Elige el par con menor SCE de prueba, sin regla de empate."""
    return min(filas, key=lambda f: f.sce_prueba)


def precision_linea_de_base(
    R: np.ndarray,
    M_ent: np.ndarray,
    M_prueba: np.ndarray,
    semilla_inicializacion: int,
    escala_inicializacion: float,
    epsilon: float,
    max_iter: int,
    top_n: int,
    umbral_relevante: float,
) -> float:
    """Precisión@N del modelo de main sobre Ω_ent.

    `entrenar_als` con k = `config.K_DEFECTO` (2), λ = 0 y sin sesgos, desde
    `semilla_inicializacion`; ordenamiento absoluto (U·Vᵀ).
    """
    m, n = R.shape
    U0, V0 = inicializar_factores(
        m=m, n=n, k=config.K_DEFECTO, escala=escala_inicializacion,
        generador=np.random.default_rng(semilla_inicializacion),
    )
    modelo = entrenar_als(R, M_ent, U0, V0, epsilon=epsilon, max_iter=max_iter, lambda_=0.0)
    return precision_en_n(
        predecir(modelo.U, modelo.V), M_ent, R, M_prueba, top_n, umbral_relevante
    )


# --- CLI ---


def construir_parser() -> argparse.ArgumentParser:
    """Parámetros del experimento; todos con defaults de `config.py`."""
    parser = argparse.ArgumentParser(
        prog="python -m experimentos.sesgos",
        description="Barrido de (k, λ) con ALS con sesgos sobre MovieLens.",
    )
    parser.add_argument("--datos", type=Path, default=config.RUTA_DATOS_DEFECTO)
    parser.add_argument("--umbral", type=int, default=config.UMBRAL_DEFECTO)
    parser.add_argument("--grilla-k", type=int, nargs="+", default=list(config.GRILLA_K_SESGOS))
    parser.add_argument(
        "--grilla-lambda", type=float, nargs="+", default=list(config.GRILLA_LAMBDA_SESGOS)
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
    parser.add_argument("--top-n", type=int, default=config.TOP_N_DEFECTO)
    parser.add_argument("--umbral-relevante", type=float, default=config.UMBRAL_RELEVANTE)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Carga MovieLens, parte Ω, barre (k, λ) con sesgos y reporta la tabla y el par elegido."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("src").setLevel(logging.WARNING)  # sin progreso por iteración
    args = construir_parser().parse_args(argv)

    # El filtro valida umbral >= k; se valida contra el mayor k de la grilla.
    datos = preparar_datos_movielens(
        args.datos / "ratings.csv", args.datos / "movies.csv", args.umbral, max(args.grilla_k)
    )
    R, M = datos.calificaciones.R, datos.calificaciones.M
    LOGGER.info("MovieLens filtrado (umbral=%d): %d usuarios × %d películas", args.umbral, *R.shape)
    M_ent, M_prueba = particionar(
        M, args.fraccion_prueba, np.random.default_rng(args.semilla_particion)
    )

    filas = barrer_sesgos(
        R, M, M_ent, M_prueba,
        grilla_k=tuple(args.grilla_k), grilla_lambda=tuple(args.grilla_lambda),
        semilla_inicializacion=args.semilla_inicializacion,
        escala_inicializacion=config.ESCALA_INICIALIZACION_DEFECTO,
        epsilon=args.epsilon, max_iter=args.max_iter,
        escala_min=args.escala_min, escala_max=args.escala_max,
        top_n=args.top_n, umbral_relevante=args.umbral_relevante,
    )
    elegido = elegir_par_sesgos(filas)
    precision_base = precision_linea_de_base(
        R, M_ent, M_prueba,
        semilla_inicializacion=args.semilla_inicializacion,
        escala_inicializacion=config.ESCALA_INICIALIZACION_DEFECTO,
        epsilon=args.epsilon, max_iter=args.max_iter,
        top_n=args.top_n, umbral_relevante=args.umbral_relevante,
    )

    LOGGER.info(
        "\n| k | λ | iteraciones | corte | f final | SCE de prueba | fuera de rango "
        "| max|r̂| fuera de Ω | precisión@%d A | precisión@%d B |", args.top_n, args.top_n,
    )
    LOGGER.info("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for f in filas:
        LOGGER.info(
            "| %d | %g | %d | %s | %.4f | %.4f | %.2f %% | %.2f | %.4f | %.4f |",
            f.k, f.lambda_, f.iteraciones, f.motivo_corte, f.f_final, f.sce_prueba,
            100 * f.fraccion_fuera_de_rango, f.max_abs_fuera_de_omega,
            f.precision_a, f.precision_b,
        )
    LOGGER.info(
        "\nPar elegido: k=%d, λ=%g (SCE de prueba %.4f; optimista: la selección se hizo "
        "sobre el mismo conjunto de prueba)", elegido.k, elegido.lambda_, elegido.sce_prueba,
    )
    LOGGER.info(
        "Línea de base (main: k=%d, λ=0, sin sesgos, sobre Ω_ent): precisión@%d = %.4f",
        config.K_DEFECTO, args.top_n, precision_base,
    )


if __name__ == "__main__":
    main()
