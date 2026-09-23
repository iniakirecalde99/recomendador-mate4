"""resolver_factor y entrenar_als (informe 3.5)."""

import numpy as np
import pytest

from src.als import entrenar_als, resolver_factor
from src.errores import SistemaSingularError
from src.modelo import inicializar_factores


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


def test_historial_de_sce_de_als_no_crece(matriz_ejemplo_informe, generador_fijo):
    R, M = matriz_ejemplo_informe
    U0, V0 = inicializar_factores(m=4, n=5, k=2, escala=1.0, generador=generador_fijo)

    resultado = entrenar_als(R, M, U0, V0, epsilon=1e-8, max_iter=100)

    historial = np.array(resultado.historial_f)
    assert np.all(np.diff(historial) <= 1e-9)  # no creciente, con margen numérico


def test_als_ejemplo_4x5_no_lanza_singular_y_sce_decrece(matriz_ejemplo_informe, generador_fijo):
    R, M = matriz_ejemplo_informe
    U0, V0 = inicializar_factores(m=4, n=5, k=2, escala=1.0, generador=generador_fijo)

    resultado = entrenar_als(R, M, U0, V0, epsilon=1e-8, max_iter=100)

    assert resultado.historial_f[-1] < resultado.historial_f[0]


@pytest.mark.skip(
    reason="V0 del ejemplo de la sección 5 del informe todavía no está fijado"
)
def test_als_ejemplo_4x5_primera_iteracion_coincide_con_informe():
    """Compara la primera iteración de ALS contra la tabla del informe (sección 5).

    Pendiente hasta fijar la V0 exacta que usa ese ejemplo.
    """
