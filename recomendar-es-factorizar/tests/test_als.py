"""resolver_factor (informe 3.5)."""

import numpy as np
import pytest

from src.als import resolver_factor
from src.errores import SistemaSingularError


def test_resolver_factor_transpuesto_coincide_con_calculo_manual_de_v():
    # 3 usuarios x 3 películas, k=2.
    R = np.array(
        [
            [5.0, 4.0, np.nan],
            [3.0, np.nan, 2.0],
            [1.0, 5.0, 4.0],
        ]
    )
    M = np.array(
        [
            [True, True, False],
            [True, False, True],
            [True, True, True],
        ]
    )
    U = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    V = resolver_factor(R.T, M.T, U)

    # Paso de V fila por fila a mano, sin transponer: para cada película j,
    # arma las ecuaciones normales con los usuarios que la calificaron.
    n_peliculas = R.shape[1]
    k = U.shape[1]
    V_manual = np.empty((n_peliculas, k))
    for j in range(n_peliculas):
        usuarios_obs = np.where(M[:, j])[0]
        U_obs = U[usuarios_obs, :]
        r_obs = R[usuarios_obs, j]
        V_manual[j, :] = np.linalg.solve(U_obs.T @ U_obs, U_obs.T @ r_obs)

    np.testing.assert_allclose(V, V_manual)


def test_resolver_factor_lanza_sistema_singular_si_hay_menos_datos_que_incognitas():
    # Fila 0: una sola columna observada para k=2 incógnitas -> sistema singular.
    R = np.array([[5.0, np.nan], [3.0, 4.0]])
    M = np.array([[True, False], [True, True]])
    F = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(SistemaSingularError) as exc_info:
        resolver_factor(R, M, F)

    assert exc_info.value.fila == 0
    assert exc_info.value.n_observados == 1
