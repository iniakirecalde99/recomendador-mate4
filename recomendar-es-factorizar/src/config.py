"""Valores por defecto de los hiperparámetros de la demo.

Valores definitivos, calibrados corriendo la demo y un experimento de
calibración sobre MovieLens real (tablas completas
y conclusión en experimentos/bitacora.md). El criterio de cada uno queda como
comentario debajo de la constante.
"""

from pathlib import Path

# --- Preprocesamiento y modelo ---

UMBRAL_DEFECTO: int = 50
# Motivo (calibración sobre MovieLens real, ver experimentos/bitacora.md): primer
# umbral, de 20/50/100 probados, donde ALS y GD con k=2 dan
# max|r_hat| fuera de Ω < 7 para los dos a la vez (con umbral=20 sigue
# siendo >7; con umbral=100 el filtro en cascada vacía la matriz). El
# umbral es un parámetro propio, independiente de k (>= k, si no
# `filtrar_por_minimo` lanza `UmbralInsuficienteError`): tener al menos k
# observaciones por fila alcanza para que el sistema de ALS tenga solución
# única, pero no para que esa solución sea razonable.

K_DEFECTO: int = 2
# Motivo (calibración sobre MovieLens real): de los k probados (2, 3, 5)
# con umbral=50, k=2 es el que menos amplifica |r_hat| fuera de Ω (5.8 vs.
# 6.5 y 11.9) y evita el SistemaSingularError que aparece con
# umbral == k == 5.

ETA_DEFECTO: float = 2e-4
# Motivo (calibración sobre MovieLens real, k=2): de los eta probados
# (2e-4, 5e-4, 1e-3), 2e-4 converge por tolerancia sin que f aumente; 5e-4
# no converge en 3000 iteraciones (f sube 1491 veces) y 1e-3 diverge
# (DivergenciaError) en la iteración 14.

EPSILON_DEFECTO: float = 1.0
# Motivo: con la escala de la SCE sobre MovieLens filtrado (decenas de
# miles), 1.0 es chico para no cortar apenas arranca y grande para que ALS
# y GD corten por tolerancia en unas pocas decenas/cientos de iteraciones
# en vez de agotar max_iter.

MAX_ITER_DEFECTO: int = 3000
# Motivo: GD con eta=2e-4 necesitó hasta ~1800 iteraciones para cortar por
# tolerancia en la calibración (umbral=20, k=5); 3000 deja margen sin
# alargar demasiado una demo en vivo.

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

RUTA_DATOS_DEFECTO: Path = Path("data/ml-100k")
# Carpeta con `u.data` y `u.item`, la misma que usa scripts/descargar_movielens.py.

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
