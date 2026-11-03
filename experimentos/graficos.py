"""Figuras 1 y 2 de la sección 8 del informe.

Lee el JSON que genera `experimentos/seccion8.py` y dibuja la convergencia
de f para ALS y descenso de gradiente (Figura 1) y la dispersión de las
filas de V de ALS para las películas más calificadas (Figura 2). Números con
coma decimal y punto de miles, como en el informe. No modifica nada de `src/`.

Uso: `python -m experimentos.graficos --help`
"""

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # solo se guarda a archivo

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from experimentos.seccion8 import RUTA_RESULTADOS_DEFECTO

LOGGER = logging.getLogger(__name__)

RUTA_FIGURAS_DEFECTO: Path = Path("experimentos/figuras")
AZUL, NARANJA = "#2a78d6", "#eb6834"  # primeras dos series de la paleta categórica
TEXTO, TEXTO_SECUNDARIO, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"
ITERACIONES_RECUADRO: int = 30


def _miles(x: float, _posicion: object = None) -> str:
    """Entero con punto de miles: 100000 -> '100.000'."""
    return f"{x:,.0f}".replace(",", ".")


def _coma(x: float, _posicion: object = None) -> str:
    """Un decimal con coma: 0.5 -> '0,5'."""
    return f"{x:.1f}".replace(".", ",")


def _sin_etiqueta(_x: float, _posicion: object = None) -> str:
    return ""


def _estilo(ax: plt.Axes) -> None:
    """Ejes recesivos: sin bordes superior y derecho, grilla clara detrás."""
    for borde in ("top", "right"):
        ax.spines[borde].set_visible(False)
    ax.grid(True, color=GRILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def _titulo_corto(titulo: str) -> str:
    """Acorta un título de MovieLens para la etiqueta del gráfico.

    Saca el año y los "a.k.a.", pasa "Godfather, The" a "The Godfather" y
    abrevia Star Wars y El señor de los anillos.
    """
    titulo = re.sub(r"\s*\(a\.k\.a\.[^)]*\)", "", titulo)
    titulo = re.sub(r"\s*\((Indiana Jones[^)]*)\)", "", titulo)
    articulo = re.match(r"(.*?), (The|A)( \(\d{4}\))$", titulo)
    if articulo:
        titulo = f"{articulo.group(2)} {articulo.group(1)}{articulo.group(3)}"
    titulo = re.sub(r" \(\d{4}\)$", "", titulo)
    titulo = titulo.replace("Star Wars: Episode ", "Star Wars ").replace("Lord of the Rings: The ", "LOTR: ")
    for subtitulo in (" - A New Hope", " - The Empire Strikes Back", " - Return of the Jedi"):
        titulo = titulo.replace(subtitulo, "")
    return titulo


def graficar_convergencia(resultados: dict, ruta_salida: Path) -> None:
    """Figura 1: f por iteración para ALS y GD, escala log, con recuadro de las primeras iteraciones."""
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    for metodo, color in (("ALS", AZUL), ("GD", NARANJA)):
        historial = resultados[f"hist_{metodo}"]
        ultimo = len(historial) - 1
        ax.plot(range(len(historial)), historial, color=color, linewidth=2,
                marker="o" if metodo == "ALS" else None, markersize=4)
        nombre = "ALS" if metodo == "ALS" else "Descenso de gradiente"
        ax.annotate(f"{nombre}: {ultimo} iteraciones", xy=(ultimo, historial[-1]),
                    xytext=(ultimo + (8 if metodo == "ALS" else -10),
                            historial[-1] * (1.9 if metodo == "ALS" else 1.5)),
                    ha="left" if metodo == "ALS" else "right", color=TEXTO, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=TEXTO_SECUNDARIO, lw=0.8))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_miles))
    ax.yaxis.set_minor_formatter(FuncFormatter(_sin_etiqueta))
    ax.set_yticks([20000, 50000, 100000, 200000, 350000])

    recuadro = ax.inset_axes([0.42, 0.42, 0.55, 0.5])
    for metodo, color in (("ALS", AZUL), ("GD", NARANJA)):
        historial = resultados[f"hist_{metodo}"][: ITERACIONES_RECUADRO + 1]
        recuadro.plot(range(len(historial)), historial, color=color, linewidth=2, marker="o", markersize=3)
    recuadro.set_yscale("log")
    recuadro.set_ylim(15000, 400000)
    recuadro.set_yticks([20000, 50000, 100000, 300000])
    recuadro.yaxis.set_major_formatter(FuncFormatter(_miles))
    recuadro.yaxis.set_minor_formatter(FuncFormatter(_sin_etiqueta))
    recuadro.set_title(f"primeras {ITERACIONES_RECUADRO} iteraciones", fontsize=9, color=TEXTO_SECUNDARIO,
                       backgroundcolor="white")
    recuadro.tick_params(labelsize=8)
    _estilo(recuadro)

    ax.set_xlabel("iteración t")
    ax.set_ylabel("f(U, V) sobre Ω de entrenamiento (escala log)")
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta_salida)
    plt.close(fig)


def graficar_dispersion(resultados: dict, ruta_salida: Path) -> None:
    """Figura 2: filas de V (ALS, k = 2) de las películas más calificadas, con su título.

    Cada etiqueta prueba posiciones alrededor de su punto, de la más cercana a
    la más lejana, y se queda con la primera cuya caja (medida con el
    renderer) no pisa otra etiqueta ya puesta, ningún punto ni el borde de los
    ejes. Las que quedan lejos de su punto llevan una línea guía.
    """
    puntos = resultados["dispersion"]
    x = np.array([p["v1"] for p in puntos])
    y = np.array([p["v2"] for p in puntos])
    etiquetas = [_titulo_corto(p["titulo"]) for p in puntos]

    fig, ax = plt.subplots(figsize=(9, 9), dpi=200)
    ax.scatter(x, y, s=28, color=AZUL, edgecolor="white", linewidth=1, zorder=3)
    ax.set_xlabel("primera coordenada de vⱼ")
    ax.set_ylabel("segunda coordenada de vⱼ")
    ax.xaxis.set_major_formatter(FuncFormatter(_coma))
    ax.yaxis.set_major_formatter(FuncFormatter(_coma))
    ax.set_xlim(x.min() - 0.12, x.max() + 0.3)
    ax.set_ylim(y.min() - 0.08, y.max() + 0.08)
    _estilo(ax)
    fig.tight_layout()

    renderer = fig.canvas.get_renderer()
    marco = ax.get_window_extent(renderer)
    puntos_pantalla = ax.transData.transform(np.column_stack([x, y]))
    radio = 4 * fig.dpi / 72  # medio diámetro del marcador, con margen
    # Desplazamientos candidatos en puntos tipográficos: (dx, dy, alineación horizontal)
    candidatos = [(6, 0, "left"), (-6, 0, "right")]
    for distancia in (9, 14, 20, 27, 35, 44):
        candidatos += [(6, distancia, "left"), (6, -distancia, "left"),
                       (-6, distancia, "right"), (-6, -distancia, "right")]
    puestas: list = []

    def choca(caja) -> bool:
        if (caja.x0 < marco.x0 or caja.x1 > marco.x1 or caja.y0 < marco.y0 or caja.y1 > marco.y1):
            return True
        if any(caja.overlaps(otra) for otra in puestas):
            return True
        return any(caja.x0 - radio < px < caja.x1 + radio and caja.y0 - radio < py < caja.y1 + radio
                   for px, py in puntos_pantalla)

    # Primero las zonas más densas, que tienen menos lugares libres.
    vecinos = [np.sum(np.hypot(*(puntos_pantalla - p).T) < 60) for p in puntos_pantalla]
    for i in sorted(range(len(puntos)), key=lambda i: -vecinos[i]):
        for dx, dy, alineacion in candidatos:
            texto = ax.annotate(etiquetas[i], xy=(x[i], y[i]), xytext=(dx, dy), textcoords="offset points",
                                fontsize=7, color=TEXTO, va="center", ha=alineacion)
            caja = texto.get_window_extent(renderer).expanded(1.04, 1.15)
            if not choca(caja):
                break
            texto.remove()
        else:
            LOGGER.warning("sin lugar libre para la etiqueta %r; queda en la última posición", etiquetas[i])
            texto = ax.annotate(etiquetas[i], xy=(x[i], y[i]), xytext=(dx, dy), textcoords="offset points",
                                fontsize=7, color=TEXTO, va="center", ha=alineacion)
            caja = texto.get_window_extent(renderer)
        if dy != 0:
            ax.annotate("", xy=(x[i], y[i]), xytext=(dx, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#a8a7a2", lw=0.5, shrinkA=0, shrinkB=3))
        puestas.append(caja)

    fig.savefig(ruta_salida)
    plt.close(fig)


def construir_parser() -> argparse.ArgumentParser:
    """Arma el parser de `python -m experimentos.graficos`."""
    parser = argparse.ArgumentParser(prog="python -m experimentos.graficos",
                                     description="Figuras 1 y 2 de la sección 8 del informe.")
    parser.add_argument("--resultados", type=Path, default=RUTA_RESULTADOS_DEFECTO)
    parser.add_argument("--figuras", type=Path, default=RUTA_FIGURAS_DEFECTO)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Lee los resultados y guarda las dos figuras."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = construir_parser().parse_args(argv)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": TEXTO_SECUNDARIO, "axes.labelcolor": TEXTO,
                         "xtick.color": TEXTO_SECUNDARIO, "ytick.color": TEXTO_SECUNDARIO,
                         "font.family": "DejaVu Sans"})
    resultados = json.loads(args.resultados.read_text(encoding="utf-8"))
    args.figuras.mkdir(parents=True, exist_ok=True)
    graficar_convergencia(resultados, args.figuras / "figura_convergencia.png")
    graficar_dispersion(resultados, args.figuras / "figura_dispersion_V.png")
    LOGGER.info("figuras guardadas en %s", args.figuras)


if __name__ == "__main__":
    main()
