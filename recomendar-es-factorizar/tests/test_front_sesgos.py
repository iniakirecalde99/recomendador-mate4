"""Recomendador HTML con sesgos.

Los tests de paridad ejecutan el JS de la página con node (el bloque
`<script id="modelo">`, sin DOM) y lo comparan con Python.
"""

import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from experimentos.front_sesgos import generar_html_sesgos
from src import config
from src.front import peliculas_mas_calificadas
from src.modelo import inicializar_factores
from src.sesgos import calcular_mu, entrenar_als_sesgos, paso_usuarios

NODE = shutil.which("node")
requiere_node = pytest.mark.skipif(
    NODE is None, reason="requiere node para ejecutar el JS de la página"
)

K, LAMBDA = 3, 2.0


@pytest.fixture(scope="module")
def modelo_chico():
    generador = np.random.default_rng(5)
    m, n = 30, 40
    M = generador.uniform(size=(m, n)) < 0.6
    R = np.where(M, generador.integers(1, 11, size=(m, n)) / 2, np.nan)
    U0, V0 = inicializar_factores(m=m, n=n, k=K, escala=1.0, generador=np.random.default_rng(7))
    modelo = entrenar_als_sesgos(
        R, M, U0, V0, mu=calcular_mu(R, M), epsilon=1e-8, max_iter=200, lambda_=LAMBDA
    )
    titulos = {j: f"Película {j} (2000)" for j in range(n)}
    return modelo, M, titulos


@pytest.fixture(scope="module")
def pagina(modelo_chico, tmp_path_factory):
    modelo, M, titulos = modelo_chico
    ruta = tmp_path_factory.mktemp("front") / "recomendador-sesgos.html"
    generar_html_sesgos(modelo, LAMBDA, titulos, M, ruta, top_n=10)
    return ruta.read_text(encoding="utf-8")


def _datos(html):
    texto = re.search(r'<script type="application/json" id="datos">(.*?)</script>', html, re.S)
    return json.loads(texto.group(1))


def _js_modelo(html):
    return re.search(r'<script id="modelo">(.*?)</script>', html, re.S).group(1)


def _correr_js(html, tmp_path, cuerpo, datos=None):
    """Ejecuta el JS del modelo con node y devuelve lo que imprime `cuerpo` como JSON."""
    datos = _datos(html) if datos is None else datos
    programa = (
        f"const datos = {json.dumps(datos)};\n{_js_modelo(html)}\n{cuerpo}\n"
    )
    archivo = tmp_path / "prueba.js"
    archivo.write_text(programa, encoding="utf-8")
    salida = subprocess.run([NODE, str(archivo)], capture_output=True, text=True, check=True)
    return json.loads(salida.stdout)


# --- Exportación ---


def test_generar_html_exporta_mu_v_c_k_lambda_y_titulos(modelo_chico, pagina):
    modelo, M, titulos = modelo_chico
    datos = _datos(pagina)

    assert datos["mu"] == modelo.mu
    assert np.array_equal(np.array(datos["V"]), modelo.V)
    assert np.array_equal(np.array(datos["c"]), modelo.c)
    assert datos["k"] == K
    assert datos["lambda"] == LAMBDA
    assert datos["titulos"] == [titulos[j] for j in range(len(titulos))]
    # las mismas 30 películas que el recomendador actual
    assert datos["a_calificar"] == peliculas_mas_calificadas(M, config.N_A_CALIFICAR_DEFECTO)
    assert datos["top_n"] == 10
    assert datos["min_calificaciones"] == config.MIN_CALIFICACIONES_FRONT_DEFECTO


def test_config_tiene_el_par_elegido_en_ts08():
    assert config.K_FRONT_SESGOS == 20
    assert config.LAMBDA_FRONT_SESGOS == 10.0


# --- JS del modelo, ejecutado con node ---


@requiere_node
def test_solver_del_js_resuelve_con_pivoteo_parcial(pagina, tmp_path):
    # A[0][0] = 0: sin pivoteo parcial la eliminación divide por cero.
    A = [[0.0, 2.0, 1.0], [1.0, 1.0, 0.0], [3.0, 0.0, 1.0]]
    b = [5.0, 3.0, 4.0]

    x = _correr_js(
        pagina, tmp_path,
        f"console.log(JSON.stringify(resolverSistema({json.dumps(A)}, {json.dumps(b)})));",
    )

    np.testing.assert_allclose(x, np.linalg.solve(np.array(A), np.array(b)), rtol=1e-12)


def _usuarios_fijos(datos):
    a = datos["a_calificar"]
    return [
        {a[0]: 5, a[1]: 4, a[2]: 3, a[3]: 2, a[4]: 1},
        {a[5]: 5, a[6]: 5, a[7]: 1, a[8]: 2, a[9]: 4, a[10]: 3, a[11]: 5, a[12]: 1},
        {a[2]: 5, a[4]: 5, a[6]: 5, a[8]: 5, a[10]: 5, a[12]: 5},
    ]


def _python_usuario(modelo, calificaciones, lambda_, top_n):
    n = modelo.V.shape[0]
    r = np.full(n, np.nan)
    for j, nota in calificaciones.items():
        r[j] = nota
    u, b = paso_usuarios(r[None, :], ~np.isnan(r)[None, :], modelo.V, modelo.c, modelo.mu, lambda_)
    u, b = u[0], float(b[0])
    personal = modelo.V @ u
    completo = modelo.mu + b + modelo.c + personal
    tops = {}
    for modo, puntajes in (("A", completo), ("B", personal)):
        puntajes = puntajes.copy()
        puntajes[list(calificaciones)] = -np.inf
        tops[modo] = np.argsort(-puntajes, kind="stable")[:top_n].tolist()
    return u, b, tops


_CUERPO_PARIDAD = """
const usuarios = %s;
const salida = usuarios.map((calif) => {
  const calificaciones = new Map(calif.map(([j, r]) => [j, r]));
  const vector = resolverUsuario(datos, calificaciones);
  const tops = {};
  for (const modo of ["A", "B"]) {
    tops[modo] = topN(datos, vector.u, vector.b, new Set(calificaciones.keys()), modo)
      .map(([j]) => j);
  }
  return { u: vector.u, b: vector.b, tops };
});
console.log(JSON.stringify(salida));
"""


@requiere_node
def test_paridad_js_python_para_tres_usuarios(modelo_chico, pagina, tmp_path):
    modelo, _, _ = modelo_chico
    usuarios = _usuarios_fijos(_datos(pagina))

    resultado_js = _correr_js(
        pagina, tmp_path,
        _CUERPO_PARIDAD % json.dumps([list(u.items()) for u in usuarios]),
    )

    for calificaciones, js in zip(usuarios, resultado_js):
        u, b, tops = _python_usuario(modelo, calificaciones, LAMBDA, top_n=10)
        assert np.allclose(js["u"], u) and np.allclose(js["b"], b)
        assert js["tops"]["A"] == tops["A"]
        assert js["tops"]["B"] == tops["B"]
        assert not set(js["tops"]["A"]) & set(calificaciones)  # sin las calificadas


@requiere_node
def test_el_js_toma_lambda_del_json(modelo_chico, pagina, tmp_path):
    modelo, _, _ = modelo_chico
    datos = _datos(pagina)
    datos["lambda"] = 0.5  # otro λ en el JSON: el JS tiene que usarlo
    usuarios = _usuarios_fijos(datos)[:1]

    resultado_js = _correr_js(
        pagina, tmp_path,
        _CUERPO_PARIDAD % json.dumps([list(u.items()) for u in usuarios]),
        datos=datos,
    )

    u, b, _ = _python_usuario(modelo, usuarios[0], 0.5, top_n=10)
    u_lambda_original, _, _ = _python_usuario(modelo, usuarios[0], LAMBDA, top_n=10)
    assert np.allclose(resultado_js[0]["u"], u) and np.allclose(resultado_js[0]["b"], b)
    assert not np.allclose(u, u_lambda_original)
