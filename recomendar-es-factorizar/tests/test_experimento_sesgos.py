"""Precisión@10, barrido, reentrenamiento y
criterio del experimento de sesgos."""

import numpy as np
import pytest

import experimentos.sesgos as experimento
from experimentos.sesgos import (
    FilaSesgos,
    barrer_sesgos,
    construir_parser,
    elegir_par_sesgos,
    precision_linea_de_base,
    precision_en_n,
)
from src import config
from src.datos import particionar
from src.modelo import inicializar_factores
from src.sesgos import calcular_mu


# --- Precisión@N ---


def _caso_precision():
    # 3 usuarios × 5 películas. Puntajes ordenan 0 > 1 > 2 > 3 > 4 para todos.
    puntajes = np.tile(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), (3, 1))
    R = np.array(
        [
            [3.0, 4.5, 2.0, np.nan, 4.0],
            [5.0, 1.0, np.nan, 4.0, np.nan],
            [2.0, 3.0, 3.5, np.nan, 1.0],
        ]
    )
    M_ent = np.array(
        [
            [True, False, False, False, False],  # usuario 0: la 0 está en entrenamiento
            [False, True, False, False, False],
            [True, True, False, False, False],
        ]
    )
    M_prueba = ~np.isnan(R) & ~M_ent
    return puntajes, M_ent, R, M_prueba


def test_precision_en_n_calculada_a_mano():
    # top-2:
    # usuario 0: sin la 0 (entrenamiento), top-2 = {1, 2}; relevantes en
    #   prueba: 1 (4,5) y 4 (4,0) → acierta 1 → 1/2.
    # usuario 1: sin la 1, top-2 = {0, 2}; relevantes: 0 (5,0) y 3 (4,0) → 1/2.
    # usuario 2: en prueba tiene 2 (3,5) y 4 (1,0): ninguna relevante → afuera.
    puntajes, M_ent, R, M_prueba = _caso_precision()

    precision = precision_en_n(puntajes, M_ent, R, M_prueba, top_n=2, umbral_relevante=4.0)

    assert precision == pytest.approx((0.5 + 0.5) / 2)


def test_precision_excluye_del_top_lo_que_esta_en_entrenamiento():
    puntajes, M_ent, R, M_prueba = _caso_precision()
    # Si la película 0 del usuario 1 pasa a entrenamiento, deja de contar
    # como acierto y sale de su top.
    M_ent = M_ent.copy()
    M_ent[1, 0] = True
    M_prueba = M_prueba.copy()
    M_prueba[1, 0] = False

    precision = precision_en_n(puntajes, M_ent, R, M_prueba, top_n=2, umbral_relevante=4.0)

    # usuario 0: 1/2; usuario 1: top-2 = {2, 3}, relevante la 3 → 1/2.
    assert precision == pytest.approx(0.5)


def test_relevante_es_mayor_o_igual_que_el_umbral():
    puntajes = np.array([[2.0, 1.0]])
    M_ent = np.zeros((1, 2), dtype=bool)
    M_prueba = np.ones((1, 2), dtype=bool)

    con_cuatro = precision_en_n(
        puntajes, M_ent, np.array([[4.0, 1.0]]), M_prueba, top_n=1, umbral_relevante=4.0
    )
    con_tres_y_medio = precision_en_n(
        puntajes, M_ent, np.array([[3.5, 4.5]]), M_prueba, top_n=1, umbral_relevante=4.0
    )

    assert con_cuatro == 1.0  # 4,0 cuenta como relevante
    assert con_tres_y_medio == 0.0  # 3,5 no; el usuario cuenta por el 4,5 que no está en su top-1


def test_config_tiene_grilla_y_umbrales_del_experimento():
    assert config.UMBRAL_RELEVANTE == 4.0
    assert config.GRILLA_K_SESGOS == (2, 5, 10, 20)
    assert config.GRILLA_LAMBDA_SESGOS == (1.0, 5.0, 10.0, 20.0)
    assert config.MAX_FRECUENCIA_SESGOS == 0.25
    assert config.MIN_PELICULAS_DISTINTAS_SESGOS == 127
    assert config.MAX_FRACCION_FUERA_DE_RANGO_SESGOS == 0.01
    # las de la regularización no se tocan
    assert config.GRILLA_K == (2, 3, 5, 10)
    assert config.MIN_PELICULAS_EXITO == 130


# --- Barrido, selección y línea de base ---


def _datos_chicos():
    generador = np.random.default_rng(3)
    m, n = 14, 18
    M = generador.uniform(size=(m, n)) < 0.75
    R = np.where(M, generador.integers(1, 11, size=(m, n)) / 2, np.nan)
    M_ent, M_prueba = particionar(M, fraccion_prueba=0.25, generador=np.random.default_rng(1))
    return R, M, M_ent, M_prueba


def _barrer(R, M, M_ent, M_prueba, grilla_k=(1, 2), grilla_lambda=(1.0, 5.0)):
    return barrer_sesgos(
        R, M, M_ent, M_prueba,
        grilla_k=grilla_k, grilla_lambda=grilla_lambda,
        semilla_inicializacion=7, escala_inicializacion=1.0,
        epsilon=1e-6, max_iter=200, escala_min=0.5, escala_max=5.0,
        top_n=3, umbral_relevante=4.0,
    )


def test_barrido_devuelve_una_fila_por_par_con_precision_a_y_b():
    filas = _barrer(*_datos_chicos())

    assert [(f.k, f.lambda_) for f in filas] == [(1, 1.0), (1, 5.0), (2, 1.0), (2, 5.0)]
    for fila in filas:
        assert fila.iteraciones > 0
        assert np.isfinite(fila.sce_prueba) and fila.sce_prueba > 0
        assert 0.0 <= fila.precision_a <= 1.0 and 0.0 <= fila.precision_b <= 1.0
        assert 0.0 <= fila.fraccion_fuera_de_rango <= 1.0


def test_barrido_usa_mu_de_omega_ent(monkeypatch):
    R, M, M_ent, M_prueba = _datos_chicos()
    mus = []
    entrenar_real = experimento.entrenar_als_sesgos

    def entrenar_espia(R_, M_, U0, V0, mu, epsilon, max_iter, lambda_):
        mus.append((mu, M_.copy()))
        return entrenar_real(R_, M_, U0, V0, mu=mu, epsilon=epsilon, max_iter=max_iter, lambda_=lambda_)

    monkeypatch.setattr(experimento, "entrenar_als_sesgos", entrenar_espia)

    _barrer(R, M, M_ent, M_prueba, grilla_k=(2,), grilla_lambda=(1.0,))

    (mu, M_usada), = mus
    assert mu == calcular_mu(R, M_ent) != calcular_mu(R, M)
    assert np.array_equal(M_usada, M_ent)


def _fila(k, lambda_, sce_prueba):
    return FilaSesgos(
        k=k, lambda_=lambda_, iteraciones=5, motivo_corte="tolerancia", f_final=0.0,
        sce_prueba=sce_prueba, fraccion_fuera_de_rango=0.0, max_abs_fuera_de_omega=0.0,
        precision_a=0.1, precision_b=0.1,
    )


def test_seleccion_elige_la_menor_sce_de_prueba_sin_empate():
    # (10, 5) está a 0,1 % de (20, 1): sin regla de empate gana la menor igual.
    filas = [_fila(2, 1.0, 1000.0), _fila(10, 5.0, 801.0), _fila(20, 1.0, 800.0)]

    assert elegir_par_sesgos(filas) is filas[2]


def test_linea_de_base_usa_entrenar_als_de_main_sobre_omega_ent(monkeypatch):
    R, M, M_ent, M_prueba = _datos_chicos()
    llamadas = []
    entrenar_real = experimento.entrenar_als

    def entrenar_espia(R_, M_, U0, V0, epsilon, max_iter, lambda_=0.0):
        llamadas.append((M_.copy(), U0.copy(), V0.copy(), epsilon, max_iter, lambda_))
        return entrenar_real(R_, M_, U0, V0, epsilon=epsilon, max_iter=max_iter, lambda_=lambda_)

    monkeypatch.setattr(experimento, "entrenar_als", entrenar_espia)

    precision = precision_linea_de_base(
        R, M_ent, M_prueba, semilla_inicializacion=7, escala_inicializacion=1.0,
        epsilon=1e-6, max_iter=200, top_n=3, umbral_relevante=4.0,
    )

    (M_usada, U0, V0, epsilon, max_iter, lambda_), = llamadas
    assert np.array_equal(M_usada, M_ent)
    assert U0.shape[1] == config.K_DEFECTO == 2
    assert lambda_ == 0.0
    U_esperada, V_esperada = inicializar_factores(
        m=R.shape[0], n=R.shape[1], k=2, escala=1.0, generador=np.random.default_rng(7)
    )
    assert np.array_equal(U0, U_esperada) and np.array_equal(V0, V_esperada)
    assert (epsilon, max_iter) == (1e-6, 200)
    assert 0.0 <= precision <= 1.0


def test_parser_toma_los_defaults_de_config():
    args = construir_parser().parse_args([])

    assert args.datos == config.RUTA_DATOS_DEFECTO
    assert args.umbral == config.UMBRAL_DEFECTO
    assert tuple(args.grilla_k) == config.GRILLA_K_SESGOS
    assert tuple(args.grilla_lambda) == config.GRILLA_LAMBDA_SESGOS
    assert args.fraccion_prueba == config.FRACCION_PRUEBA
    assert args.semilla_particion == config.SEMILLA_PARTICION
    assert args.semilla_inicializacion == config.SEMILLA_INICIALIZACION
    assert args.epsilon == config.EPSILON_DEFECTO
    assert args.max_iter == config.MAX_ITER_DEFECTO
    assert args.escala_min == config.ESCALA_MIN
    assert args.escala_max == config.ESCALA_MAX
    assert args.top_n == config.TOP_N_DEFECTO
    assert args.umbral_relevante == config.UMBRAL_RELEVANTE
