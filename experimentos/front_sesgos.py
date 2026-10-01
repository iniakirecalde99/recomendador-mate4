"""Recomendador HTML con sesgos.

`python -m experimentos.front_sesgos` reentrena con ALS sobre todo Ω el par
elegido en el experimento de sesgos (k = 20, λ = 10, en `config.py`) y genera una página
estática, sin servidor ni dependencias externas, con μ, V, c, k, λ y los
títulos incrustados como JSON (el mismo mecanismo que `src/front.py`). En el
navegador, el usuario califica algunas de las 30 películas más calificadas y
la página resuelve su (u, b) con el paso aumentado, sin
volver a entrenar, y muestra el top-10 con el ordenamiento A o B.

No modifica `src/front.py` ni `recomendador.html`: reusa la selección de
películas, los géneros y la escala de `src/front.py`.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from experimentos.sesgos import reentrenar_sesgos_sobre_todo_omega
from src import config
from src.datos import cargar_generos, preparar_datos_movielens
from src.front import ETIQUETAS_CALIFICACION, GENEROS_EN_ESPANOL, peliculas_mas_calificadas
from src.sesgos import ResultadoSesgos

LOGGER = logging.getLogger(__name__)


def generar_html_sesgos(
    modelo: ResultadoSesgos,
    lambda_: float,
    titulos_por_indice: dict[int, str],
    M: np.ndarray,
    ruta_salida: Path,
    generos_por_indice: dict[int, list[str]] | None = None,
    n_a_calificar: int = config.N_A_CALIFICAR_DEFECTO,
    top_n: int = config.TOP_N_DEFECTO,
    min_calificaciones: int = config.MIN_CALIFICACIONES_FRONT_DEFECTO,
) -> None:
    """Escribe la página del recomendador con sesgos, con el modelo como JSON incrustado.

    El JSON lleva μ, V, c, k, λ, los títulos (y géneros), las películas para
    calificar (las `n_a_calificar` con más calificaciones en M, las mismas
    que el recomendador de main), `top_n` y `min_calificaciones`. El JS no
    tiene ningún parámetro del modelo escrito a mano: todo sale del JSON.
    """
    n = modelo.V.shape[0]
    datos = {
        "mu": modelo.mu,
        "V": modelo.V.tolist(),
        "c": modelo.c.tolist(),
        "k": int(modelo.V.shape[1]),
        "lambda": lambda_,
        "titulos": [titulos_por_indice[j] for j in range(n)],
        "generos": [
            [GENEROS_EN_ESPANOL.get(g, g) for g in (generos_por_indice or {}).get(j, [])]
            for j in range(n)
        ],
        "a_calificar": peliculas_mas_calificadas(M, n_a_calificar),
        "top_n": top_n,
        "min_calificaciones": min_calificaciones,
        "etiquetas_calificacion": ETIQUETAS_CALIFICACION,
    }
    # "<\/" es un escape válido en JSON y evita que un título cierre el <script>.
    json_incrustado = json.dumps(datos, ensure_ascii=True).replace("</", "<\\/")

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(_PLANTILLA_HTML.replace("__DATOS__", json_incrustado), encoding="utf-8")
    LOGGER.info("Recomendador con sesgos guardado en %s", ruta_salida)


def construir_parser() -> argparse.ArgumentParser:
    """Parámetros del generador; todos con defaults de `config.py`."""
    parser = argparse.ArgumentParser(
        prog="python -m experimentos.front_sesgos",
        description="Genera el recomendador HTML con sesgos (par elegido en el experimento) para un usuario nuevo.",
    )
    parser.add_argument("--datos", type=Path, default=config.RUTA_DATOS_DEFECTO)
    parser.add_argument("--salida", type=Path, default=config.RUTA_FRONT_SESGOS_DEFECTO)
    parser.add_argument("--umbral", type=int, default=config.UMBRAL_DEFECTO)
    parser.add_argument("--k", type=int, default=config.K_FRONT_SESGOS)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=config.LAMBDA_FRONT_SESGOS)
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_DEFECTO)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER_DEFECTO)
    parser.add_argument(
        "--semilla-inicializacion", type=int, default=config.SEMILLA_INICIALIZACION
    )
    parser.add_argument("--n-a-calificar", type=int, default=config.N_A_CALIFICAR_DEFECTO)
    parser.add_argument("--top-n", type=int, default=config.TOP_N_DEFECTO)
    parser.add_argument(
        "--min-calificaciones", type=int, default=config.MIN_CALIFICACIONES_FRONT_DEFECTO
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Reentrena el par elegido sobre todo Ω y genera la página."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = construir_parser().parse_args(argv)

    datos = preparar_datos_movielens(
        args.datos / "ratings.csv", args.datos / "movies.csv", args.umbral, args.k
    )
    R, M = datos.calificaciones.R, datos.calificaciones.M
    # El par elegido, reentrenado con ALS sobre todo Ω (μ de todo Ω).
    modelo = reentrenar_sesgos_sobre_todo_omega(
        R, M, k=args.k, lambda_=args.lambda_,
        semilla_inicializacion=args.semilla_inicializacion,
        escala_inicializacion=config.ESCALA_INICIALIZACION_DEFECTO,
        epsilon=args.epsilon, max_iter=args.max_iter,
    )
    LOGGER.info(
        "Modelo con sesgos (k=%d, λ=%g) sobre todo Ω: μ=%.4f, %d it (%s)",
        args.k, args.lambda_, modelo.mu, modelo.n_iteraciones, modelo.motivo_corte,
    )

    generos_por_id = cargar_generos(args.datos / "movies.csv")
    generos_por_indice = {
        indice: generos_por_id[id_pelicula]
        for id_pelicula, indice in datos.calificaciones.id_pelicula_a_indice.items()
    }
    generar_html_sesgos(
        modelo, args.lambda_, datos.titulos_por_indice, M, args.salida,
        generos_por_indice=generos_por_indice, n_a_calificar=args.n_a_calificar,
        top_n=args.top_n, min_calificaciones=args.min_calificaciones,
    )


_PLANTILLA_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recomendador con sesgos</title>
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
  .estrellas button:focus-visible, .modos input:focus-visible { outline: 2px solid var(--acento); outline-offset: 2px; }
  .escala { color: var(--tenue); margin: 0 0 8px; }
  .escala b { color: var(--texto); font-variant-numeric: tabular-nums; }
  .generos { display: block; color: var(--tenue); font-size: 0.85em; }
  .modos { display: flex; flex-direction: column; gap: 6px; margin: 0 0 12px; border: 0; padding: 0; }
  .modos label { display: flex; gap: 8px; align-items: baseline; cursor: pointer; }
  .modos .ayuda { color: var(--tenue); font-size: 0.85em; }
  #estado { color: var(--tenue); margin: 0 0 12px; }
  #estado.aviso { color: var(--aviso); }
  ol#recomendaciones { margin: 0; padding-left: 1.6em; }
  ol#recomendaciones li { padding: 4px 0; }
  .prediccion { color: var(--tenue); font-variant-numeric: tabular-nums; white-space: nowrap; }
  #vector { font-family: ui-monospace, Consolas, monospace; color: var(--tenue); margin-top: 12px; font-size: 0.85em; }
</style>
</head>
<body>
<main>
  <h1>Recomendador con sesgos</h1>
  <p class="bajada">Calificá algunas películas de MovieLens. La página calcula tu gusto
    (u) y tu sesgo (b) con el modelo con sesgos, sin volver a entrenar, y te recomienda
    películas que no calificaste.</p>
  <div class="columnas">
    <section>
      <h2>Las más calificadas de MovieLens</h2>
      <p class="escala">Calificá de 1 a 5 las que viste; las que no, dejalas en blanco.</p>
      <p class="escala" id="escala"></p>
      <ul id="a-calificar"></ul>
    </section>
    <section aria-live="polite">
      <h2>Tus recomendaciones</h2>
      <fieldset class="modos">
        <legend class="escala">Ordenar por:</legend>
        <label><input type="radio" name="modo" value="A" checked>
          <span>A: estimación completa <span class="ayuda">(μ + b + c + u·V: favorece a las
          películas bien valoradas por todos)</span></span></label>
        <label><input type="radio" name="modo" value="B">
          <span>B: afinidad personal <span class="ayuda">(u·V: sin el sesgo de la
          película)</span></span></label>
      </fieldset>
      <p id="estado"></p>
      <ol id="recomendaciones"></ol>
      <p id="vector"></p>
    </section>
  </div>
</main>
<script type="application/json" id="datos">__DATOS__</script>
<script id="modelo">
"use strict";
// Modelo con sesgos. Todo parámetro sale de `datos`: μ, V,
// c, k y λ los exporta experimentos/front_sesgos.py.

// Eliminación gaussiana con pivoteo parcial para A x = b (A de n × n).
// Devuelve null si el sistema es singular.
function resolverSistema(A, b) {
  const n = b.length;
  const M = A.map((fila, i) => [...fila, b[i]]);
  let escala = 0;
  for (const fila of A) for (const v of fila) escala = Math.max(escala, Math.abs(v));
  for (let col = 0; col < n; col++) {
    let pivote = col;
    for (let f = col + 1; f < n; f++) {
      if (Math.abs(M[f][col]) > Math.abs(M[pivote][col])) pivote = f;
    }
    if (!(Math.abs(M[pivote][col]) > 1e-12 * escala)) return null;
    [M[col], M[pivote]] = [M[pivote], M[col]];
    for (let f = col + 1; f < n; f++) {
      const factor = M[f][col] / M[col][col];
      for (let c = col; c <= n; c++) M[f][c] -= factor * M[col][c];
    }
  }
  const x = new Array(n).fill(0);
  for (let f = n - 1; f >= 0; f--) {
    let suma = M[f][n];
    for (let c = f + 1; c < n; c++) suma -= M[f][c] * x[c];
    x[f] = suma / M[f][f];
  }
  return x;
}

// Paso de usuarios aumentado para el usuario nuevo. Incógnitas
// (u, b) ∈ R^(k+1), matriz [V | 1] de las películas calificadas, objetivo
// r − μ − c; ecuaciones normales ([V | 1]ᵀ [V | 1] + λ I) x = [V | 1]ᵀ y,
// igual que resolver_factor en src/als.py.
function resolverUsuario(datos, calificaciones) {
  const k = datos.k;
  const n1 = k + 1;
  const A = Array.from({ length: n1 }, () => new Array(n1).fill(0));
  const rhs = new Array(n1).fill(0);
  for (const [j, r] of calificaciones) {
    const fila = [...datos.V[j], 1];
    const y = r - datos.mu - datos.c[j];
    for (let a = 0; a < n1; a++) {
      rhs[a] += fila[a] * y;
      for (let b = 0; b < n1; b++) A[a][b] += fila[a] * fila[b];
    }
  }
  for (let a = 0; a < n1; a++) A[a][a] += datos.lambda;
  const x = resolverSistema(A, rhs);
  return x === null ? null : { u: x.slice(0, k), b: x[k] };
}

// Top-N sin las calificadas. A ordena por r̂ completo
// (μ + b + cⱼ + Vⱼ·u); B, por la parte personal (Vⱼ·u). Devuelve
// [índice, r̂ completo], con empates por índice (como en Python).
function topN(datos, u, b, calificadas, modo) {
  const candidatos = [];
  for (let j = 0; j < datos.V.length; j++) {
    if (calificadas.has(j)) continue;
    let personal = 0;
    for (let a = 0; a < datos.k; a++) personal += datos.V[j][a] * u[a];
    const completo = datos.mu + b + datos.c[j] + personal;
    candidatos.push([j, modo === "A" ? completo : personal, completo]);
  }
  candidatos.sort((x, y) => y[1] - x[1]);
  return candidatos.slice(0, datos.top_n).map(([j, , completo]) => [j, completo]);
}
</script>
<script>
"use strict";
const datos = JSON.parse(document.getElementById("datos").textContent);
const calificaciones = new Map();  // índice de película -> calificación 1..5

function modoElegido() {
  return document.querySelector('input[name="modo"]:checked').value;
}

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

function actualizar() {
  const estado = document.getElementById("estado");
  const lista = document.getElementById("recomendaciones");
  const vector = document.getElementById("vector");
  lista.replaceChildren();
  vector.textContent = "";
  estado.classList.remove("aviso");

  const faltan = datos.min_calificaciones - calificaciones.size;
  if (faltan > 0) {
    estado.textContent = `Calificá al menos ${datos.min_calificaciones} películas para ` +
      `calcular tu gusto (te ${faltan === 1 ? "falta 1" : "faltan " + faltan}).`;
    return;
  }
  const resultado = resolverUsuario(datos, calificaciones);
  if (resultado === null) {
    estado.classList.add("aviso");
    estado.textContent = "Con estas calificaciones el sistema es singular: calificá alguna película más.";
    return;
  }
  const modo = modoElegido();
  estado.textContent = `Top-${datos.top_n} con el ordenamiento ${modo}, entre las películas ` +
    `que no calificaste (${calificaciones.size} calificaciones).`;
  for (const [j, rHat] of topN(datos, resultado.u, resultado.b, new Set(calificaciones.keys()), modo)) {
    const li = document.createElement("li");
    li.appendChild(nodoPelicula(j));
    const pred = document.createElement("span");
    pred.className = "prediccion";
    pred.textContent = ` r̂ = ${rHat.toFixed(2)}`;
    li.appendChild(pred);
    lista.appendChild(li);
  }
  vector.textContent = `b = ${resultado.b.toFixed(3)} · μ = ${datos.mu.toFixed(3)} · ` +
    `k = ${datos.k} · λ = ${datos.lambda}`;
}

function armarEscala() {
  const escala = document.getElementById("escala");
  datos.etiquetas_calificacion.forEach((etiqueta, i) => {
    if (i > 0) escala.append(" · ");
    const numero = document.createElement("b");
    numero.textContent = i + 1;
    escala.append(numero, " = " + etiqueta);
  });
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
      boton.title = `${r} = ${datos.etiquetas_calificacion[r - 1]}`;
      boton.setAttribute("aria-label", boton.title);
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

for (const radio of document.querySelectorAll('input[name="modo"]')) {
  radio.addEventListener("change", actualizar);
}
armarEscala();
armarListaACalificar();
actualizar();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
