"""Página estática del recomendador (usuario nuevo con V de ALS fija)."""

import json
import re

import numpy as np
import pytest

from src import front
from src.als import resolver_factor
from src.errores import DimensionLatenteNoSoportadaError


def _extraer_json_incrustado(html: str) -> dict:
    coincidencia = re.search(
        r'<script type="application/json" id="datos">(.*?)</script>', html, re.DOTALL
    )
    assert coincidencia is not None, "el HTML no tiene el bloque JSON con los datos"
    return json.loads(coincidencia.group(1))


def _formula_2x2_del_js(V_obs: np.ndarray, r_obs: np.ndarray) -> np.ndarray:
    """Réplica en Python de la cuenta que hace el JS: ecuaciones normales 2×2 por Cramer."""
    a = float(np.sum(V_obs[:, 0] * V_obs[:, 0]))
    b = float(np.sum(V_obs[:, 0] * V_obs[:, 1]))
    d = float(np.sum(V_obs[:, 1] * V_obs[:, 1]))
    e = float(np.sum(V_obs[:, 0] * r_obs))
    f = float(np.sum(V_obs[:, 1] * r_obs))
    det = a * d - b * b
    return np.array([(d * e - b * f) / det, (a * f - b * e) / det])


def test_generar_html_crea_el_archivo_con_v_y_titulos_en_json(tmp_path):
    V = np.array([[1.0, 0.5], [0.2, 1.3], [0.7, 0.7]])
    titulos_por_indice = {0: "Toy Story (1995)", 1: "Película </script> rara", 2: "Nixon (1995)"}
    M = np.array([[True, True, False], [True, False, True], [True, True, True]])
    ruta = tmp_path / "salidas" / "recomendador.html"

    front.generar_html(V, titulos_por_indice, M, ruta, n_a_calificar=2, top_n=1)

    assert ruta.exists()
    datos = _extraer_json_incrustado(ruta.read_text(encoding="utf-8"))
    assert np.allclose(np.array(datos["V"]), V)
    assert datos["titulos"] == [titulos_por_indice[j] for j in range(3)]
    assert datos["k"] == 2
    assert datos["top_n"] == 1
    # las 2 más calificadas: columna 0 (3 calificaciones) y después 1 o 2 (2 cada una)
    assert datos["a_calificar"][0] == 0
    assert len(datos["a_calificar"]) == 2


def test_peliculas_mas_calificadas_ordena_por_cantidad_de_calificaciones():
    M = np.array(
        [
            [True, False, True, True],
            [False, False, True, True],
            [True, False, True, False],
        ]
    )

    # empates (columnas 0 y 3, con 2 cada una) se desempatan por índice
    assert front.peliculas_mas_calificadas(M, cantidad=3) == [2, 0, 3]
    assert front.peliculas_mas_calificadas(M, cantidad=1) == [2]


def test_generar_html_rechaza_k_distinto_de_2(tmp_path):
    V = np.ones((3, 3))
    M = np.ones((2, 3), dtype=bool)

    with pytest.raises(DimensionLatenteNoSoportadaError):
        front.generar_html(V, {0: "a", 1: "b", 2: "c"}, M, tmp_path / "x.html")


def test_vector_de_usuario_con_resolver_factor_coincide_con_formula_2x2_del_js():
    # Caso fijo: 5 películas, el usuario nuevo califica 3 (índices 0, 2 y 3).
    V = np.array([[1.2, 0.3], [0.4, 1.1], [0.9, 0.8], [0.1, 1.5], [1.0, 1.0]])
    r_usuario = np.array([5.0, np.nan, 4.0, 2.0, np.nan])
    m_usuario = ~np.isnan(r_usuario)

    # Informe 5: paso de U de ALS con V fija, para una sola fila.
    u_als = resolver_factor(r_usuario[None, :], m_usuario[None, :], V)[0]
    u_js = _formula_2x2_del_js(V[m_usuario], r_usuario[m_usuario])

    assert np.allclose(u_als, u_js)
