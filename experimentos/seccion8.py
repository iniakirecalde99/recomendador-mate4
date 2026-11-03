"""Experimento de la sección 8 del informe: entrenamiento, prueba y comparación.

Con MovieLens filtrado (defaults de `config.py`), arma la tabla de pasadas
del filtro, divide las calificaciones en entrenamiento y prueba por usuario,
entrena ALS y descenso de gradiente sobre entrenamiento y calcula RMSE,
líneas de base, el barrido de k, el top-5 de un usuario y los datos de la
dispersión de V. Guarda todo en un JSON que lee `experimentos/graficos.py`.
No modifica nada de `src/`.

Uso: `python -m experimentos.seccion8 --help`
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src import config
from src.als import entrenar_als
from src.datos import (
    cargar_calificaciones,
    cargar_generos,
    cargar_titulos,
    construir_indice_a_titulo,
    filtrar_por_minimo,
)
from src.errores import DivergenciaError, SistemaSingularError
from src.gradiente import entrenar_gd
from src.modelo import inicializar_factores, predecir, sce

LOGGER = logging.getLogger(__name__)

PROPORCION_PRUEBA_DEFECTO: float = 0.2
KS_BARRIDO_DEFECTO: tuple[int, ...] = (2, 5, 10)
N_POPULARES_DEFECTO: int = 40  # películas de la dispersión de V (sección 8.4)
N_TOP_DEFECTO: int = 5
MIN_PELICULAS_GENERO: int = 15  # géneros con menos películas no se correlacionan
RUTA_RESULTADOS_DEFECTO: Path = Path("salidas/seccion8.json")


def dividir_por_usuario(
    M: np.ndarray, proporcion_prueba: float, k: int, generador: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Divide Ω en entrenamiento y prueba, fila por fila.

    De cada usuario elige al azar round(proporcion_prueba · |Ωᵢ|) de sus
    calificaciones para prueba, pero nunca tantas que le queden menos de k en
    entrenamiento (si no, su sistema de ALS sería singular). Devuelve
    (M_ent, M_pru), disjuntas y con unión M.
    """
    M_ent = M.copy()
    M_pru = np.zeros_like(M)
    for i in range(M.shape[0]):
        observadas = np.where(M[i])[0]
        n_prueba = min(int(round(proporcion_prueba * len(observadas))), max(len(observadas) - k, 0))
        elegidas = generador.choice(observadas, size=n_prueba, replace=False)
        M_ent[i, elegidas] = False
        M_pru[i, elegidas] = True
    return M_ent, M_pru


def rmse(R: np.ndarray, R_hat: np.ndarray, mascara: np.ndarray) -> float:
    """RMSE = √(SCE / |mascara|), solo sobre las posiciones True de `mascara`."""
    error = np.where(mascara, R - R_hat, 0.0)
    return float(np.sqrt((error**2).sum() / mascara.sum()))


def pasadas_del_filtro(M: np.ndarray, umbral: int) -> list[dict]:
    """Usuarios, películas, calificaciones y |Ω|/(m·n) después de cada pasada.

    Repite el mismo criterio que `filtrar_por_minimo`; la pasada 0 es la
    matriz sin filtrar.
    """
    filas = np.arange(M.shape[0])
    columnas = np.arange(M.shape[1])
    pasadas = []
    n_pasada = 0
    while True:
        sub = M[np.ix_(filas, columnas)]
        pasadas.append(dict(pasada=n_pasada, usuarios=len(filas), peliculas=len(columnas),
                            calificaciones=int(sub.sum()),
                            densidad=float(sub.sum() / (len(filas) * len(columnas)))))
        nuevas_filas = filas[sub.sum(axis=1) >= umbral]
        nuevas_columnas = columnas[sub.sum(axis=0) >= umbral]
        if len(nuevas_filas) == len(filas) and len(nuevas_columnas) == len(columnas):
            return pasadas
        filas, columnas = nuevas_filas, nuevas_columnas
        n_pasada += 1


def construir_parser() -> argparse.ArgumentParser:
    """Arma el parser de `python -m experimentos.seccion8`."""
    parser = argparse.ArgumentParser(
        prog="python -m experimentos.seccion8",
        description="Experimentos de la sección 8 del informe sobre MovieLens.",
    )
    parser.add_argument("--datos", type=Path, default=config.RUTA_DATOS_DEFECTO)
    parser.add_argument("--salida", type=Path, default=RUTA_RESULTADOS_DEFECTO)
    parser.add_argument("--umbral", type=int, default=config.UMBRAL_DEFECTO)
    parser.add_argument("--k", type=int, default=config.K_DEFECTO)
    parser.add_argument("--eta", type=float, default=config.ETA_DEFECTO)
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_DEFECTO)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER_DEFECTO)
    parser.add_argument("--semilla", type=int, default=config.SEMILLA_DEFECTO)
    parser.add_argument("--proporcion-prueba", type=float, default=PROPORCION_PRUEBA_DEFECTO)
    parser.add_argument("--usuario", type=int, default=config.USUARIO_DEFECTO)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Corre todos los cálculos de la sección 8 y guarda el JSON de resultados."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("src").setLevel(logging.WARNING)  # sin el progreso por iteración
    args = construir_parser().parse_args(argv)

    res: dict = {}

    # 8.1: pasadas del filtro
    crudo = cargar_calificaciones(args.datos / "ratings.csv")
    res["pasadas"] = pasadas_del_filtro(crudo.M, args.umbral)
    datos = filtrar_por_minimo(crudo, args.umbral, args.k)
    R, M = datos.R, datos.M
    m, n = R.shape
    titulos = construir_indice_a_titulo(datos.id_pelicula_a_indice, cargar_titulos(args.datos / "movies.csv"))
    generos_por_id = cargar_generos(args.datos / "movies.csv")
    generos = {j: generos_por_id[id_crudo] for id_crudo, j in datos.id_pelicula_a_indice.items()}
    res["final"] = dict(m=m, n=n, omega=int(M.sum()), variables=(m + n) * args.k)
    res["crudo_peliculas_menos_umbral"] = int((crudo.M.sum(axis=0) < args.umbral).sum())
    res["fraccion_calif_conservadas"] = int(M.sum()) / int(crudo.M.sum())
    # Películas y usuarios que llegaban al umbral en el crudo pero cayeron en pasadas siguientes
    conservadas = set(datos.id_pelicula_a_indice)
    res["peliculas_umbral_crudo_eliminadas"] = sum(
        1 for id_crudo, j in crudo.id_pelicula_a_indice.items()
        if crudo.M[:, j].sum() >= args.umbral and id_crudo not in conservadas)
    conservados = set(datos.id_usuario_a_indice)
    res["usuarios_umbral_crudo_eliminados"] = sum(
        1 for id_crudo, i in crudo.id_usuario_a_indice.items()
        if crudo.M[i].sum() >= args.umbral and id_crudo not in conservados)

    # 8.2: división por usuario
    M_ent, M_pru = dividir_por_usuario(M, args.proporcion_prueba, args.k, np.random.default_rng(args.semilla))
    R_ent = np.where(M_ent, R, np.nan)
    res["division"] = dict(
        entrenamiento=int(M_ent.sum()), prueba=int(M_pru.sum()),
        min_ent_por_usuario=int(M_ent.sum(axis=1).min()),
        min_ent_por_pelicula=int(M_ent.sum(axis=0).min()),
        peliculas_sin_prueba=int((M_pru.sum(axis=0) == 0).sum()),
    )

    def evaluar(R_hat: np.ndarray) -> dict:
        return dict(
            rmse_ent=rmse(R, R_hat, M_ent),
            rmse_pru=rmse(R, R_hat, M_pru),
            rmse_pru_recortado=rmse(R, np.clip(R_hat, 0.5, 5.0), M_pru),
            max_abs_fuera=float(np.abs(R_hat[~M]).max()),
            frac_fuera_rango_pru=float(((R_hat < 0.5) | (R_hat > 5))[M_pru].mean()),
        )

    def sce_ent(R_hat: np.ndarray) -> float:
        return float((np.where(M_ent, R - R_hat, 0.0) ** 2).sum())

    # Líneas de base
    mu = float(np.nanmean(R_ent))
    base_global = np.full((m, n), mu)
    base_usuario = np.repeat(np.nanmean(R_ent, axis=1)[:, None], n, axis=1)
    res["base_global"] = dict(mu=mu, sce_ent=sce_ent(base_global), **evaluar(base_global))
    res["base_usuario"] = dict(sce_ent=sce_ent(base_usuario), **evaluar(base_usuario))

    def entrenar(k: int, metodo: str):
        generador = np.random.default_rng(args.semilla)
        U0, V0 = inicializar_factores(m, n, k, config.ESCALA_INICIALIZACION_DEFECTO, generador)
        try:
            if metodo == "ALS":
                r = entrenar_als(R_ent, M_ent, U0.copy(), V0.copy(), args.epsilon, args.max_iter)
            else:
                r = entrenar_gd(R_ent, M_ent, U0.copy(), V0.copy(), args.eta, args.epsilon, args.max_iter)
        except (DivergenciaError, SistemaSingularError) as error:
            return None, dict(error=f"{type(error).__name__}: {error}")
        return r, dict(iter=r.n_iteraciones, corte=r.motivo_corte, f0=r.historial_f[0],
                       f_final=r.historial_f[-1], tiempo=r.tiempo_segundos,
                       **evaluar(predecir(r.U, r.V)))

    # 8.3: ALS y descenso de gradiente con k por defecto
    modelos = {}
    for metodo in ("ALS", "GD"):
        r, resumen = entrenar(args.k, metodo)
        modelos[metodo] = r
        res[metodo] = resumen
        res[f"hist_{metodo}"] = r.historial_f
        assert abs(sce(R_ent, M_ent, r.U, r.V) - r.historial_f[-1]) < 1e-6  # f final = SCE sobre Ω_E
        resumen["rmse_ent_desde_f"] = float(np.sqrt(r.historial_f[-1] / M_ent.sum()))  # √(f/|Ω_E|)

    res["barrido_k"] = {k: {metodo: entrenar(k, metodo)[1] for metodo in ("ALS", "GD")}
                        for k in KS_BARRIDO_DEFECTO}

    # Top-5 de un usuario, entre las películas que no calificó (en todo Ω)
    i_usuario = datos.id_usuario_a_indice[args.usuario]
    res["usuario"] = dict(id=args.usuario, n_calif=int(M[i_usuario].sum()),
                          media=float(np.nanmean(R[i_usuario])))
    res["top5"] = {}
    for metodo, r in modelos.items():
        fila = predecir(r.U, r.V)[i_usuario]
        candidatas = [j for j in np.argsort(-fila) if not M[i_usuario, j]][:N_TOP_DEFECTO]
        res["top5"][metodo] = [(titulos[j], float(fila[j]), generos[j]) for j in candidatas]

    # 8.4: V de ALS. Los géneros solo se usan para interpretar.
    V = modelos["ALS"].V
    populares = np.argsort(-M.sum(axis=0))[:N_POPULARES_DEFECTO]
    res["dispersion"] = [dict(titulo=titulos[j], v1=float(V[j, 0]), v2=float(V[j, 1]),
                              n=int(M[:, j].sum()), generos=generos[j]) for j in populares]
    todos_generos = sorted({g for gs in generos.values() for g in gs})
    res["corr_genero"] = {}
    for eje in range(V.shape[1]):
        for genero in todos_generos:
            indicador = np.array([genero in generos[j] for j in range(n)], dtype=float)
            if indicador.sum() >= MIN_PELICULAS_GENERO:
                res["corr_genero"].setdefault(eje, {})[genero] = float(np.corrcoef(indicador, V[:, eje])[0, 1])
    res["extremos_todas"] = {}
    for eje in range(V.shape[1]):
        orden = np.argsort(V[:, eje])
        res["extremos_todas"][eje] = dict(
            menores=[(titulos[j], float(V[j, eje]), generos[j]) for j in orden[:8]],
            mayores=[(titulos[j], float(V[j, eje]), generos[j]) for j in orden[::-1][:8]],
        )

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    LOGGER.info("resultados guardados en %s", args.salida)


if __name__ == "__main__":
    main()
