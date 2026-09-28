"""Recomendador HTML para un usuario nuevo, con la V de ALS fija (informe 5).

`python -m src.front` entrena ALS sobre MovieLens y genera una página
estática (sin servidor ni dependencias externas) con V y los títulos
incrustados como JSON. En el navegador, el usuario califica algunas de las
películas más calificadas y la página calcula su vector u con el mismo paso
de U de ALS (ecuaciones normales con V fija), resuelto con la fórmula
cerrada del sistema 2×2.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src import config
from src.als import entrenar_als
from src.datos import cargar_generos, preparar_datos_movielens
from src.errores import DimensionLatenteNoSoportadaError, MinimoCalificacionesInsuficienteError
from src.modelo import inicializar_factores

LOGGER = logging.getLogger(__name__)

K_SOPORTADO = 2  # el JS resuelve el sistema 2×2 con fórmula cerrada

# Nombres de `u.genre` de MovieLens → cómo se muestran en la página.
GENEROS_EN_ESPANOL: dict[str, str] = {
    "unknown": "sin género",
    "Action": "acción",
    "Adventure": "aventura",
    "Animation": "animación",
    "Children's": "infantil",
    "Comedy": "comedia",
    "Crime": "crimen",
    "Documentary": "documental",
    "Drama": "drama",
    "Fantasy": "fantasía",
    "Film-Noir": "cine negro",
    "Horror": "terror",
    "Musical": "musical",
    "Mystery": "misterio",
    "Romance": "romance",
    "Sci-Fi": "ciencia ficción",
    "Thriller": "suspenso",
    "War": "bélica",
    "Western": "western",
}


def peliculas_mas_calificadas(M: np.ndarray, cantidad: int) -> list[int]:
    """Índices de las `cantidad` columnas de M con más calificaciones (empates por índice)."""
    conteos = M.sum(axis=0)
    orden = np.argsort(-conteos, kind="stable")
    return [int(j) for j in orden[:cantidad]]


def generar_html(
    V: np.ndarray,
    titulos_por_indice: dict[int, str],
    M: np.ndarray,
    ruta_salida: Path,
    n_a_calificar: int = config.N_A_CALIFICAR_DEFECTO,
    top_n: int = config.TOP_N_DEFECTO,
    min_calificaciones: int = config.MIN_CALIFICACIONES_FRONT_DEFECTO,
    generos_por_indice: dict[int, list[str]] | None = None,
) -> None:
    """Escribe en `ruta_salida` la página del recomendador con V y títulos como JSON.

    La página ofrece para calificar (de 1 a 5) las `n_a_calificar` películas
    con más calificaciones en M. Con al menos `min_calificaciones`
    calificaciones resuelve el vector del usuario nuevo con V fija y muestra
    los títulos de las `top_n` películas no calificadas con mayor r̂, en
    orden y sin el valor estimado. Lanza `DimensionLatenteNoSoportadaError` si V
    no tiene exactamente 2 columnas, y `MinimoCalificacionesInsuficienteError`
    si `min_calificaciones` < k; en ambos casos, antes de escribir el archivo.

    Si se pasa `generos_por_indice` (nombres de MovieLens), la página muestra
    los géneros de cada película en español; solo son para mostrar.
    """
    k = V.shape[1]
    if k != K_SOPORTADO:
        raise DimensionLatenteNoSoportadaError(k)
    if min_calificaciones < k:
        raise MinimoCalificacionesInsuficienteError(min_calificaciones, k)

    datos = {
        "k": k,
        "top_n": top_n,
        "min_calificaciones": min_calificaciones,
        "V": V.tolist(),
        "titulos": [titulos_por_indice[j] for j in range(V.shape[0])],
        "a_calificar": peliculas_mas_calificadas(M, n_a_calificar),
        "generos": [
            [GENEROS_EN_ESPANOL.get(g, g) for g in (generos_por_indice or {}).get(j, [])]
            for j in range(V.shape[0])
        ],
    }
    # "<\/" es un escape válido en JSON y evita que un título cierre el <script>.
    json_incrustado = json.dumps(datos, ensure_ascii=True).replace("</", "<\\/")

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(_PLANTILLA_HTML.replace("__DATOS__", json_incrustado), encoding="utf-8")
    LOGGER.info("Recomendador HTML guardado en %s", ruta_salida)


def construir_parser() -> argparse.ArgumentParser:
    """Arma el parser de `python -m src.front` (defaults de `config.py`).

    Sin `--k`: la página solo soporta k=2, así que usa `config.K_DEFECTO`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.front",
        description="Genera el recomendador HTML (V de ALS sobre MovieLens) para un usuario nuevo.",
    )
    parser.add_argument("--datos", type=Path, default=config.RUTA_DATOS_DEFECTO)
    parser.add_argument("--salida", type=Path, default=config.RUTA_FRONT_DEFECTO)
    parser.add_argument("--umbral", type=int, default=config.UMBRAL_DEFECTO)
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_DEFECTO)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER_DEFECTO)
    parser.add_argument("--semilla", type=int, default=config.SEMILLA_DEFECTO)
    parser.add_argument("--n-a-calificar", type=int, default=config.N_A_CALIFICAR_DEFECTO)
    parser.add_argument("--top-n", type=int, default=config.TOP_N_DEFECTO)
    parser.add_argument(
        "--min-calificaciones", type=int, default=config.MIN_CALIFICACIONES_FRONT_DEFECTO
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entrena ALS sobre MovieLens y genera la página del recomendador."""
    logging.basicConfig(level=logging.INFO)
    args = construir_parser().parse_args(argv)
    k = config.K_DEFECTO

    datos = preparar_datos_movielens(args.datos / "u.data", args.datos / "u.item", args.umbral, k)
    R = datos.calificaciones.R
    M = datos.calificaciones.M
    m, n = R.shape

    generador = np.random.default_rng(args.semilla)
    U0, V0 = inicializar_factores(
        m=m, n=n, k=k, escala=config.ESCALA_INICIALIZACION_DEFECTO, generador=generador
    )
    resultado = entrenar_als(R, M, U0, V0, epsilon=args.epsilon, max_iter=args.max_iter)

    generos_por_id = cargar_generos(args.datos / "u.item", args.datos / "u.genre")
    generos_por_indice = {
        indice: generos_por_id[id_pelicula]
        for id_pelicula, indice in datos.calificaciones.id_pelicula_a_indice.items()
    }

    generar_html(
        resultado.V,
        datos.titulos_por_indice,
        M,
        args.salida,
        n_a_calificar=args.n_a_calificar,
        top_n=args.top_n,
        min_calificaciones=args.min_calificaciones,
        generos_por_indice=generos_por_indice,
    )


_PLANTILLA_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recomendar es factorizar</title>
<style>
  :root {
    --fondo: #fafaf8; --superficie: #ffffff; --texto: #1d1d1b; --tenue: #6b6b66;
    --borde: #e2e1dc; --acento: #2f5d8a; --acento-texto: #ffffff; --aviso: #8a5a00;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --fondo: #16171a; --superficie: #1f2024; --texto: #ececea; --tenue: #9a9a95;
      --borde: #34353a; --acento: #7fb0e0; --acento-texto: #10141a; --aviso: #e0b050;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--fondo); color: var(--texto);
    font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  h2 { font-size: 1.05rem; margin: 0 0 12px; }
  p.bajada { color: var(--tenue); margin: 0 0 24px; }
  .columnas { display: grid; grid-template-columns: 1fr; gap: 24px; }
  @media (min-width: 820px) { .columnas { grid-template-columns: 3fr 2fr; align-items: start; } }
  section {
    background: var(--superficie); border: 1px solid var(--borde);
    border-radius: 10px; padding: 16px;
  }
  ul { list-style: none; margin: 0; padding: 0; }
  li.pelicula {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 8px; padding: 8px 0; border-top: 1px solid var(--borde);
  }
  li.pelicula:first-child { border-top: none; }
  .estrellas { display: flex; gap: 4px; }
  .estrellas button {
    width: 32px; height: 32px; border-radius: 6px; cursor: pointer;
    border: 1px solid var(--borde); background: transparent; color: var(--texto);
    font: inherit; font-variant-numeric: tabular-nums;
  }
  .estrellas button[aria-pressed="true"] {
    background: var(--acento); border-color: var(--acento); color: var(--acento-texto);
  }
  .estrellas button:focus-visible { outline: 2px solid var(--acento); outline-offset: 2px; }
  .generos { display: block; color: var(--tenue); font-size: 0.85em; }
  #estado { color: var(--tenue); margin: 0 0 12px; }
  #estado.aviso { color: var(--aviso); }
  ol#recomendaciones { margin: 0; padding-left: 1.6em; }
  ol#recomendaciones li { padding: 4px 0; }
  #vector { font-family: ui-monospace, Consolas, monospace; color: var(--tenue); margin-top: 12px; }
</style>
</head>
<body>
<main>
  <h1>Recomendar es factorizar</h1>
  <p class="bajada">Calificá algunas películas de MovieLens y la página calcula tu vector
    de gustos con la V que aprendió ALS, sin volver a entrenar.</p>
  <div class="columnas">
    <section>
      <h2>Las más calificadas de MovieLens</h2>
      <ul id="a-calificar"></ul>
    </section>
    <section aria-live="polite">
      <h2>Tus recomendaciones</h2>
      <p id="estado"></p>
      <ol id="recomendaciones"></ol>
      <p id="vector"></p>
    </section>
  </div>
</main>
<script type="application/json" id="datos">__DATOS__</script>
<script>
"use strict";
const datos = JSON.parse(document.getElementById("datos").textContent);
const V = datos.V;
const calificaciones = new Map();  // índice de película -> calificación 1..5

// Informe 5: paso de U de ALS con V fija. Para el usuario nuevo se arman las
// ecuaciones normales (V_Ωᵀ V_Ω) u = V_Ωᵀ r sobre las películas calificadas,
// igual que resolver_factor en src/als.py, y el sistema 2×2 se resuelve con
// la fórmula cerrada (regla de Cramer).
function resolverVectorUsuario() {
  let a = 0, b = 0, d = 0, e = 0, f = 0;
  for (const [j, r] of calificaciones) {
    const v0 = V[j][0], v1 = V[j][1];
    a += v0 * v0; b += v0 * v1; d += v1 * v1;
    e += v0 * r;  f += v1 * r;
  }
  const det = a * d - b * b;
  if (!Number.isFinite(det) || Math.abs(det) <= 1e-12 * Math.max(1, a * d)) {
    return null;  // sistema singular
  }
  return [(d * e - b * f) / det, (a * f - b * e) / det];
}

// Título con los géneros de MovieLens debajo (solo para mostrar: el modelo no los usa).
function nodoPelicula(j) {
  const contenedor = document.createElement("span");
  contenedor.textContent = datos.titulos[j];
  if (datos.generos[j].length > 0) {
    const generos = document.createElement("span");
    generos.className = "generos";
    generos.textContent = datos.generos[j].join(", ");
    contenedor.appendChild(generos);
  }
  return contenedor;
}

function recomendar(u) {
  const candidatos = [];
  for (let j = 0; j < V.length; j++) {
    if (calificaciones.has(j)) continue;
    candidatos.push([j, V[j][0] * u[0] + V[j][1] * u[1]]);  // r̂_j = V_j · u
  }
  candidatos.sort((x, y) => y[1] - x[1]);
  return candidatos.slice(0, datos.top_n);
}

function actualizar() {
  const estado = document.getElementById("estado");
  const lista = document.getElementById("recomendaciones");
  const vector = document.getElementById("vector");
  lista.replaceChildren();
  vector.textContent = "";
  estado.classList.remove("aviso");

  // Con k calificaciones el sistema ya es resoluble, pero queda mal
  // determinado: se pide el mínimo incrustado (config.MIN_CALIFICACIONES_FRONT_DEFECTO).
  const faltan = datos.min_calificaciones - calificaciones.size;
  if (faltan > 0) {
    estado.textContent = `Calificá al menos ${datos.min_calificaciones} películas para calcular tu vector ` +
      `(te ${faltan === 1 ? "falta 1" : "faltan " + faltan}).`;
    return;
  }
  const u = resolverVectorUsuario();
  if (u === null) {
    estado.classList.add("aviso");
    estado.textContent = "Con estas calificaciones el sistema es singular: calificá alguna película más.";
    return;
  }
  estado.textContent = `Top-${datos.top_n} entre las películas que no calificaste ` +
    `(${calificaciones.size} calificaciones).`;
  for (const [j] of recomendar(u)) {
    const li = document.createElement("li");
    li.appendChild(nodoPelicula(j));
    lista.appendChild(li);
  }
  vector.textContent = `u = (${u[0].toFixed(3)}, ${u[1].toFixed(3)})`;
}

function armarListaACalificar() {
  const ul = document.getElementById("a-calificar");
  for (const j of datos.a_calificar) {
    const li = document.createElement("li");
    li.className = "pelicula";
    const titulo = nodoPelicula(j);
    const estrellas = document.createElement("div");
    estrellas.className = "estrellas";
    estrellas.setAttribute("role", "group");
    estrellas.setAttribute("aria-label", "Calificación de " + datos.titulos[j]);
    for (let r = 1; r <= 5; r++) {
      const boton = document.createElement("button");
      boton.type = "button";
      boton.textContent = r;
      boton.setAttribute("aria-pressed", "false");
      boton.addEventListener("click", () => {
        // volver a tocar la misma calificación la borra
        const nueva = calificaciones.get(j) === r ? null : r;
        if (nueva === null) calificaciones.delete(j); else calificaciones.set(j, nueva);
        for (const otro of estrellas.children) {
          otro.setAttribute("aria-pressed", String(Number(otro.textContent) === nueva));
        }
        actualizar();
      });
      estrellas.appendChild(boton);
    }
    li.append(titulo, estrellas);
    ul.appendChild(li);
  }
}

armarListaACalificar();
actualizar();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
