"""Valores por defecto de los hiperparámetros de la demo.

Valores definitivos, recalibrados sobre MovieLens latest-small real con el
mismo experimento que se hizo para MovieLens 100K (tablas completas y conclusión en experimentos/bitacora.md, "Recalibración"). El
criterio de cada uno queda como comentario debajo de la constante; lo
verifica tests/test_calibracion.py.
"""

from pathlib import Path

# --- Preprocesamiento y modelo ---

UMBRAL_DEFECTO: int = 40
# Motivo (recalibración sobre MovieLens latest-small, ver experimentos/bitacora.md):
# menor umbral, de 10/20/30/35/40/50 probados, donde ALS y GD con k=2 dan
# max|r_hat| fuera de Ω < 7 para los dos a la vez (6.4 y 5.9; con
# umbral=35 ALS da 7.2, con 30 da 8.5). Deja 321 usuarios × 534 películas,
# contra 207 × 302 con umbral=50. El umbral es un parámetro propio,
# independiente de k (>= k, si no `filtrar_por_minimo` lanza
# `UmbralInsuficienteError`): tener al menos k observaciones por fila
# alcanza para que el sistema de ALS tenga solución única, pero no para que
# esa solución sea razonable.

K_DEFECTO: int = 2
# Motivo (recalibración): con todos los umbrales probados, k=3 y k=5 hacen
# que ALS amplifique |r_hat| fuera de Ω (17 a 268.701); k=2 es el único que
# cumple el criterio. Además, el recomendador HTML solo soporta k=2.

ETA_DEFECTO: float = 5e-4
# Motivo (recalibración, umbral=40, k=2): de los eta probados (2e-4, 5e-4,
# 1e-3), 5e-4 converge por tolerancia sin que f aumente en la mitad de
# iteraciones que 2e-4 (462 vs. 922), con 5 semillas más probadas sin
# ningún aumento de f; 1e-3 diverge (DivergenciaError) en la iteración 35.

EPSILON_DEFECTO: float = 1.0
# Motivo: con la escala de la SCE sobre MovieLens filtrado (decenas de
# miles), 1.0 es chico para no cortar apenas arranca y grande para que ALS
# y GD corten por tolerancia en unas pocas decenas/cientos de iteraciones
# en vez de agotar max_iter.

MAX_ITER_DEFECTO: int = 3000
# Motivo: GD con eta=5e-4 necesitó entre 430 y 755 iteraciones en la
# recalibración (6 semillas) y hasta ~2300 con eta=2e-4 en la exploración
# de umbral; 3000 deja margen sin alargar demasiado una demo en vivo.

SEMILLA_DEFECTO: int = 42
# Motivo: semilla arbitraria fija, solo para que la demo sea reproducible
# entre corridas; no hay un criterio de "mejor" semilla.

ESCALA_INICIALIZACION_DEFECTO: float = 1.0
# Motivo: escala simple (U0, V0 uniformes en [0, 1)), del mismo orden que
# las calificaciones más chicas (1 a 5), sin sobreestimar r_hat desde el
# arranque.

# --- Demo ---

TOP_N_DEFECTO: int = 10
# Motivo: lo que entra cómodo en pantalla durante
# la demo; a diferencia de los anteriores, no depende de la escala del
# dataset ni se calibra contra MovieLens real.

USUARIO_DEFECTO: int = 1
# Id crudo de usuario de MovieLens (no índice reindexado): el primer id del
# archivo. Si el --umbral pedido lo filtra, main() lanza
# UsuarioNoEncontradoError con un mensaje claro en vez de fallar con un
# error críptico de índice.

RUTA_DATOS_DEFECTO: Path = Path("data/ml-latest-small")
# Carpeta con `ratings.csv` y `movies.csv` de MovieLens latest-small, la
# misma que usa scripts/descargar_movielens.py.

RUTA_GRAFICO_DEFECTO: Path = Path("salidas/convergencia.png")
# Dónde se guarda el gráfico de convergencia: en salidas/, que va
# en .gitignore, junto con el resto de lo que genera el proyecto.

# --- Recomendador HTML ---

RUTA_FRONT_DEFECTO: Path = Path("salidas/recomendador.html")
# Página estática que genera `python -m src.front`, al lado del gráfico.

N_A_CALIFICAR_DEFECTO: int = 30
# Cantidad de películas (las más calificadas de MovieLens filtrado) que la
# página ofrece para calificar: conocidas para casi cualquiera y todavía
# cómodas de recorrer en una sola pantalla.

MIN_CALIFICACIONES_FRONT_DEFECTO: int = 5
# Motivo: con k calificaciones el sistema k×k del usuario nuevo es
# resoluble pero queda mal determinado (u interpola exactamente esos k
# datos); mismo criterio que el umbral del filtro, que exige bastante más
# que k observaciones por fila.

# --- Experimento de regularización (rama experimento/regularizacion) ---

FRACCION_PRUEBA: float = 0.2
# Fracción de Ω que va a prueba en la partición.

SEMILLA_PARTICION: int = SEMILLA_DEFECTO
# Semilla del sorteo de la partición. Por
# referencia a SEMILLA_DEFECTO, no un literal: es una constante separada de
# SEMILLA_INICIALIZACION, con su propio generador, que hoy comparte valor.

ESCALA_MIN: float = 0.5
ESCALA_MAX: float = 5.0
# Escala de MovieLens latest-small: una
# estimación está fuera de rango si cae fuera de [ESCALA_MIN − 1,
# ESCALA_MAX + 1] = [−0,5; 6].

# Criterios de éxito, fijados antes de correr
# el experimento: se pueden cambiar hasta que arranque el experimento, no después.
MIN_PELICULAS_EXITO: int = 130
# Al menos 130 películas en algún top-10 (el doble de las 65 de main).
MAX_FRECUENCIA_EXITO: float = 0.25
# La más frecuente, en el top-10 de a lo sumo el 25 % de los usuarios.
MAX_FRACCION_FUERA_DE_RANGO: float = 0.01
# A lo sumo 1 % de estimaciones fuera de rango.
TOLERANCIA_EMPATE: float = 0.01
# Regla de empate de la selección: pares a menos del 1 % de la mejor
# SCE de prueba se consideran empatados.
