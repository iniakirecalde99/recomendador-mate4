"""ALS: mínimos cuadrados alternados (informe 3.5 y 5)."""

import logging

import numpy as np

from src.errores import SistemaSingularError

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
