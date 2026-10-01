"""Métricas, criterios, barrido y
verificación del experimento de regularización."""

import numpy as np
import pytest

import experimentos.regularizacion as experimento
from experimentos.regularizacion import (
    FilaBarrido,
    barrer,
    construir_parser,
    elegir_par,
    evaluar_criterios,
    medir_fuera_de_rango,
    reentrenar_sobre_todo_omega,
    verificar_gd,
)
from src import config
from src.datos import particionar
from src.errores import DivergenciaError, SistemaSingularError


# --- Métricas ---


def test_fuera_de_rango_cuenta_solo_pares_no_observados():
    # U·Vᵀ = [[10, 1], [1, 10]]: los 10 están fuera de [−0,5; 6].
    U = np.array([[10.0, 1.0], [1.0, 10.0]])
    V = np.eye(2)
    # (0,0) observado (en Ω, sea de entrenamiento o de prueba): no cuenta.
    # (1,1) no observado: cuenta. (0,1) y (1,0) no observados, dentro del rango.
    M = np.array([[True, False], [False, False]])

    fraccion, maximo = medir_fuera_de_rango(U, V, M, escala_min=0.5, escala_max=5.0)

    assert fraccion == pytest.approx(1 / 3)
    assert maximo == pytest.approx(10.0)


def test_fuera_de_rango_usa_los_limites_menos_medio_y_seis():
    # Una película, cuatro usuarios no observados con r̂ = −0,5; 6; −0,51; 6,01.
    U = np.array([[-0.5], [6.0], [-0.51], [6.01]])
    V = np.array([[1.0]])
    M = np.zeros((4, 1), dtype=bool)

    fraccion, maximo = medir_fuera_de_rango(U, V, M, escala_min=0.5, escala_max=5.0)

    assert fraccion == pytest.approx(2 / 4)  # −0,5 y 6 están dentro; −0,51 y 6,01, fuera
    assert maximo == pytest.approx(6.01)


def test_config_tiene_la_escala_y_los_umbrales_de_la_seccion_9():
    assert config.ESCALA_MIN == 0.5
    assert config.ESCALA_MAX == 5.0
    assert config.MIN_PELICULAS_EXITO == 130
    assert config.MAX_FRECUENCIA_EXITO == 0.25
    assert config.MAX_FRACCION_FUERA_DE_RANGO == 0.01
    assert config.TOLERANCIA_EMPATE == 0.01


# --- Criterios de éxito ---


def _caso_exitoso(**cambios):
    valores = dict(
        k=5,
        sce_prueba_elegido=900.0,
        sce_prueba_k2_lambda0=1000.0,
        fraccion_fuera_de_rango=0.005,
        peliculas_en_algun_top=150,
        frecuencia_mas_frecuente=0.20,
    )
    valores.update(cambios)
    return evaluar_criterios(**valores)


def test_evaluar_criterios_exitoso_si_cumple_todo():
    veredicto = _caso_exitoso()

    assert veredicto.exitoso
    assert veredicto.criterios == {1: True, 2: True, 3: True, 4: True}


def test_evaluar_criterios_falla_si_k_es_2():
    veredicto = _caso_exitoso(k=2)

    assert not veredicto.exitoso
    assert veredicto.criterios[1] is False


def test_evaluar_criterios_falla_si_la_sce_de_prueba_no_mejora_a_k2_lambda0():
    assert _caso_exitoso(sce_prueba_elegido=1000.0).criterios[2] is False  # igual no alcanza
    assert not _caso_exitoso(sce_prueba_elegido=1001.0).exitoso


def test_evaluar_criterios_falla_con_mas_de_1_por_ciento_fuera_de_rango():
    assert _caso_exitoso(fraccion_fuera_de_rango=0.01).criterios[3] is True  # "a lo sumo"
    assert _caso_exitoso(fraccion_fuera_de_rango=0.0101).criterios[3] is False


def test_evaluar_criterios_falla_con_menos_de_130_peliculas_o_mas_de_25_por_ciento():
    assert _caso_exitoso(peliculas_en_algun_top=130).criterios[4] is True  # "al menos"
    assert _caso_exitoso(peliculas_en_algun_top=129).criterios[4] is False
    assert _caso_exitoso(frecuencia_mas_frecuente=0.25).criterios[4] is True  # "a lo sumo"
    assert _caso_exitoso(frecuencia_mas_frecuente=0.2501).criterios[4] is False


# --- Barrido y selección ---


def _datos_chicos():
    generador = np.random.default_rng(3)
    m, n = 12, 15
    M = generador.uniform(size=(m, n)) < 0.7
    R = np.where(M, generador.integers(1, 6, size=(m, n)).astype(float), np.nan)
    M_ent, M_prueba = particionar(M, fraccion_prueba=0.2, generador=np.random.default_rng(1))
    return R, M, M_ent, M_prueba


def _barrer(R, M, M_ent, M_prueba, grilla_k=(1, 2), grilla_lambda=(0.0, 1.0)):
    return barrer(
        R, M, M_ent, M_prueba,
        grilla_k=grilla_k, grilla_lambda=grilla_lambda,
        semilla_inicializacion=7, escala_inicializacion=1.0,
        epsilon=1e-6, max_iter=200, escala_min=0.5, escala_max=5.0,
    )


def test_barrido_devuelve_una_fila_por_par_de_la_grilla():
    filas = _barrer(*_datos_chicos())

    assert [(f.k, f.lambda_) for f in filas] == [(1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0)]
    for fila in filas:
        assert fila.error is None
        assert fila.iteraciones > 0
        assert np.isfinite(fila.sce_prueba) and np.isfinite(fila.f_final)
        assert 0.0 <= fila.fraccion_fuera_de_rango <= 1.0


def test_barrido_registra_el_error_de_un_par_singular_y_sigue(monkeypatch):
    entrenar_real = experimento.entrenar_als

    def entrenar_que_falla_con_k2_lambda0(R, M, U0, V0, epsilon, max_iter, lambda_):
        if U0.shape[1] == 2 and lambda_ == 0.0:
            raise SistemaSingularError(fila=3, n_observados=1)
        return entrenar_real(R, M, U0, V0, epsilon=epsilon, max_iter=max_iter, lambda_=lambda_)

    monkeypatch.setattr(experimento, "entrenar_als", entrenar_que_falla_con_k2_lambda0)

    filas = _barrer(*_datos_chicos())

    assert len(filas) == 4
    fallida = next(f for f in filas if f.k == 2 and f.lambda_ == 0.0)
    assert fallida.error is not None and "fila 3" in fallida.error
    assert all(f.error is None for f in filas if f is not fallida)


def test_todos_los_lambda_de_un_k_arrancan_del_mismo_u0_v0(monkeypatch):
    entrenar_real = experimento.entrenar_als
    inicios = {}

    def entrenar_que_registra(R, M, U0, V0, epsilon, max_iter, lambda_):
        inicios[(U0.shape[1], lambda_)] = (U0.copy(), V0.copy())
        return entrenar_real(R, M, U0, V0, epsilon=epsilon, max_iter=max_iter, lambda_=lambda_)

    monkeypatch.setattr(experimento, "entrenar_als", entrenar_que_registra)

    _barrer(*_datos_chicos(), grilla_k=(2,), grilla_lambda=(0.0, 1.0, 5.0))

    U_ref, V_ref = inicios[(2, 0.0)]
    for lambda_ in (1.0, 5.0):
        assert np.array_equal(inicios[(2, lambda_)][0], U_ref)
        assert np.array_equal(inicios[(2, lambda_)][1], V_ref)
    # y es la inicialización de la semilla recibida
    m, n = _datos_chicos()[0].shape
    U_esperada, V_esperada = experimento.inicializar_factores(
        m=m, n=n, k=2, escala=1.0, generador=np.random.default_rng(7)
    )
    assert np.array_equal(U_ref, U_esperada) and np.array_equal(V_ref, V_esperada)


def _fila(k, lambda_, sce_prueba, error=None):
    return FilaBarrido(
        k=k, lambda_=lambda_, iteraciones=10, motivo_corte="tolerancia", f_final=0.0,
        sce_prueba=sce_prueba, fraccion_fuera_de_rango=0.0, max_abs_fuera_de_omega=0.0,
        error=error,
    )


def test_seleccion_elige_menor_sce_de_prueba():
    filas = [_fila(2, 0.0, 1000.0), _fila(3, 1.0, 800.0), _fila(5, 5.0, 900.0)]

    assert elegir_par(filas, tolerancia_empate=0.01) is filas[1]


def test_seleccion_con_empate_menor_al_1_por_ciento_gana_el_menor_k():
    # 5,10 queda a 0,5 % de la mejor (3,5): empatan y gana el menor k.
    filas = [_fila(5, 10.0, 800.0), _fila(3, 5.0, 804.0), _fila(2, 0.0, 900.0)]

    assert elegir_par(filas, tolerancia_empate=0.01) is filas[1]


def test_seleccion_con_varios_pares_del_menor_k_gana_el_de_menor_sce():
    filas = [_fila(5, 1.0, 800.0), _fila(3, 1.0, 805.0), _fila(3, 5.0, 803.0)]

    assert elegir_par(filas, tolerancia_empate=0.01) is filas[2]


def test_seleccion_ignora_los_pares_con_error():
    filas = [_fila(2, 0.0, float("nan"), error="singular"), _fila(5, 1.0, 900.0)]

    assert elegir_par(filas, tolerancia_empate=0.01) is filas[1]


def test_parser_toma_los_defaults_de_config():
    args = construir_parser().parse_args([])

    assert args.datos == config.RUTA_DATOS_DEFECTO
    assert args.umbral == config.UMBRAL_DEFECTO
    assert tuple(args.grilla_k) == config.GRILLA_K
    assert tuple(args.grilla_lambda) == config.GRILLA_LAMBDA
    assert args.fraccion_prueba == config.FRACCION_PRUEBA
    assert args.semilla_particion == config.SEMILLA_PARTICION
    assert args.semilla_inicializacion == config.SEMILLA_INICIALIZACION
    assert args.epsilon == config.EPSILON_DEFECTO
    assert args.max_iter == config.MAX_ITER_DEFECTO
    assert args.escala_min == config.ESCALA_MIN
    assert args.escala_max == config.ESCALA_MAX
    assert args.tolerancia_empate == config.TOLERANCIA_EMPATE


def test_config_de_la_grilla_y_las_semillas():
    assert config.GRILLA_K == (2, 3, 5, 10)
    assert config.GRILLA_LAMBDA == (0.0, 1.0, 5.0, 10.0, 20.0)
    assert config.SEMILLA_INICIALIZACION == config.SEMILLA_DEFECTO
    assert config.SEMILLA_PARTICION == config.SEMILLA_DEFECTO


# --- Reentrenamiento sobre todo Ω y verificación con GD ---


def test_reentrenar_sobre_todo_omega_usa_el_par_elegido_y_la_mascara_completa(monkeypatch):
    R, M, M_ent, _ = _datos_chicos()
    llamadas = []
    entrenar_real = experimento.entrenar_als

    def entrenar_que_registra(R_, M_, U0, V0, epsilon, max_iter, lambda_):
        llamadas.append((M_.copy(), U0.shape[1], lambda_))
        return entrenar_real(R_, M_, U0, V0, epsilon=epsilon, max_iter=max_iter, lambda_=lambda_)

    monkeypatch.setattr(experimento, "entrenar_als", entrenar_que_registra)

    resultado = reentrenar_sobre_todo_omega(
        R, M, k=2, lambda_=5.0, semilla_inicializacion=7, escala_inicializacion=1.0,
        epsilon=1e-6, max_iter=200,
    )

    (M_usada, k_usado, lambda_usado), = llamadas
    assert np.array_equal(M_usada, M) and not np.array_equal(M_usada, M_ent)
    assert (k_usado, lambda_usado) == (2, 5.0)
    assert resultado.U.shape == (R.shape[0], 2)


def _gd_que_diverge_las_primeras(n_divergencias, etas_usadas, inicios):
    entrenar_real = experimento.entrenar_gd

    def entrenar_gd_falso(R, M, U0, V0, eta, epsilon, max_iter, lambda_):
        etas_usadas.append(eta)
        inicios.append((U0.copy(), V0.copy()))
        if len(etas_usadas) <= n_divergencias:
            raise DivergenciaError(iteracion=3, eta=eta)
        return entrenar_real(
            R, M, U0, V0, eta=eta, epsilon=epsilon, max_iter=max_iter, lambda_=lambda_
        )

    return entrenar_gd_falso


def _verificar(R, M_ent, M_prueba, max_reducciones=3):
    return verificar_gd(
        R, M_ent, M_prueba, k=2, lambda_=1.0, semilla_inicializacion=7,
        escala_inicializacion=1.0, eta=0.01, epsilon=1e-6, max_iter=50,
        max_reducciones_eta=max_reducciones,
    )


def test_verificar_gd_divide_eta_por_2_al_divergir_y_registra_cada_intento(monkeypatch):
    R, _, M_ent, M_prueba = _datos_chicos()
    etas, inicios = [], []
    monkeypatch.setattr(experimento, "entrenar_gd", _gd_que_diverge_las_primeras(2, etas, inicios))

    verificacion = _verificar(R, M_ent, M_prueba)

    assert etas == [0.01, 0.005, 0.0025]
    assert [i.eta for i in verificacion.intentos] == etas
    assert [i.divergio for i in verificacion.intentos] == [True, True, False]
    assert verificacion.intentos[0].iteracion_divergencia == 3
    assert not verificacion.fallo
    assert np.isfinite(verificacion.intentos[-1].sce_prueba)
    # cada intento arranca de la misma inicialización
    for U0, V0 in inicios[1:]:
        assert np.array_equal(U0, inicios[0][0]) and np.array_equal(V0, inicios[0][1])


def test_verificar_gd_se_rinde_tras_max_reducciones_y_registra_la_falla(monkeypatch):
    R, _, M_ent, M_prueba = _datos_chicos()
    etas, inicios = [], []
    monkeypatch.setattr(
        experimento, "entrenar_gd", _gd_que_diverge_las_primeras(99, etas, inicios)
    )

    verificacion = _verificar(R, M_ent, M_prueba, max_reducciones=3)  # no lanza

    assert len(verificacion.intentos) == 3 + 1
    assert all(i.divergio for i in verificacion.intentos)
    assert verificacion.fallo


def test_parser_toma_eta_y_max_reducciones_de_config():
    args = construir_parser().parse_args([])

    assert args.eta == config.ETA_DEFECTO
    assert args.max_reducciones_eta == config.MAX_REDUCCIONES_ETA
    assert config.MAX_REDUCCIONES_ETA == 3


# --- Corrida de punta a punta sobre un dataset chico (sin MovieLens) ---


def test_main_corre_de_punta_a_punta_sobre_dataset_chico(tmp_path, caplog):
    generador = np.random.default_rng(11)
    lineas_ratings = ["userId,movieId,rating,timestamp"]
    for usuario in range(1, 21):
        for pelicula in range(1, 26):
            if generador.uniform() < 0.8:
                nota = generador.integers(1, 11) / 2
                lineas_ratings.append(f"{usuario},{pelicula},{nota},0")
    (tmp_path / "ratings.csv").write_text("\n".join(lineas_ratings) + "\n", encoding="utf-8")
    (tmp_path / "movies.csv").write_text(
        "movieId,title,genres\n"
        + "".join(f"{j},Película {j} (2000),Drama\n" for j in range(1, 26)),
        encoding="utf-8",
    )

    with caplog.at_level("INFO"):
        experimento.main([
            "--datos", str(tmp_path), "--umbral", "5",
            "--grilla-k", "2", "3", "--grilla-lambda", "0", "1",
            "--max-iter", "50",
        ])

    assert "Par elegido" in caplog.text
    assert "Reentrenado sobre todo Ω" in caplog.text
    assert "GD (eta=" in caplog.text
    assert "Concentración (notas reales)" in caplog.text
    assert "Veredicto" in caplog.text
