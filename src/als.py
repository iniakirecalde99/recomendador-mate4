"""ALS: mínimos cuadrados alternados (informe 3.5 y 5)."""

import logging
import time

import numpy as np

from src.errores import SistemaSingularError
from src.modelo import ResultadoEntrenamiento, sce

LOGGER = logging.getLogger(__name__)


def resolver_factor(R: np.ndarray, M: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Resuelve por fila las ecuaciones normales de mínimos cuadrados (informe 3.5).

    Para cada fila i de R, toma las filas de F correspondientes a las
    columnas j con M[i, j] = True y resuelve con `np.linalg.solve` la fila i
    del factor nuevo. Llamada con (R, M, V) calcula el paso de U; llamada
    con (R.T, M.T, U) calcula el paso de V (misma función, sin duplicar
    lógica).

    Lanza `SistemaSingularError` si las ecuaciones normales de alguna fila
    resultan singulares (por ejemplo, con menos observaciones que columnas
    de F).
    """
    m = R.shape[0]
    k = F.shape[1]
    factor_nuevo = np.empty((m, k))

    for i in range(m):
        columnas_obs = np.where(M[i, :])[0]
        F_obs = F[columnas_obs, :]
        r_obs = R[i, columnas_obs]

        # Informe 3.5: ecuaciones normales de mínimos cuadrados por fila.
        A = F_obs.T @ F_obs
        b = F_obs.T @ r_obs
        try:
            factor_nuevo[i, :] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as error:
            raise SistemaSingularError(fila=i, n_observados=len(columnas_obs)) from error

    return factor_nuevo


def entrenar_als(
    R: np.ndarray,
    M: np.ndarray,
    U0: np.ndarray,
    V0: np.ndarray,
    epsilon: float,
    max_iter: int,
) -> ResultadoEntrenamiento:
    """Ejecuta ALS alternando el paso de U y V (informe 3.5 y 5).

    Parte de U0, V0 y en cada iteración alterna `resolver_factor(R, M, V)` y
    `resolver_factor(R.T, M.T, U)`, usando `modelo.sce` para calcular f,
    hasta que |f(t+1) − f(t)| < epsilon o se alcanza `max_iter`.
    """
    inicio = time.perf_counter()

    U, V = U0, V0
    f_actual = sce(R, M, U, V)
    historial_f = [f_actual]

    n_iteraciones = 0
    motivo_corte = "max_iter"
    while n_iteraciones < max_iter:
        # Informe 3.5/5: alternar el paso de U y el paso de V (misma función,
        # llamada con R.T y M.T para V).
        U = resolver_factor(R, M, V)
        V = resolver_factor(R.T, M.T, U)

        f_siguiente = sce(R, M, U, V)
        n_iteraciones += 1
        historial_f.append(f_siguiente)

        LOGGER.debug("ALS iteración %d: f=%.6g", n_iteraciones, f_siguiente)
        if n_iteraciones % 100 == 0:
            LOGGER.info("ALS iteración %d: f=%.6g", n_iteraciones, f_siguiente)

        if abs(f_siguiente - f_actual) < epsilon:
            motivo_corte = "tolerancia"
            f_actual = f_siguiente
            break

        f_actual = f_siguiente

    if motivo_corte == "max_iter":
        LOGGER.warning("ALS: se alcanzó max_iter=%d sin cortar por tolerancia", max_iter)

    LOGGER.info(
        "ALS terminó: %d iteraciones (motivo=%s), f final=%.6g",
        n_iteraciones, motivo_corte, historial_f[-1],
    )

    return ResultadoEntrenamiento(
        U=U,
        V=V,
        historial_f=historial_f,
        n_iteraciones=n_iteraciones,
        tiempo_segundos=time.perf_counter() - inicio,
        motivo_corte=motivo_corte,
    )
