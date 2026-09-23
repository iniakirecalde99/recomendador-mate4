"""Valores por defecto de los hiperparámetros de la demo.

PROVISORIO: los hiperparámetros de MovieLens de este archivo se fijaron a
mano sin correr la demo sobre MovieLens real (no hay
dataset real en este entorno — ver experimentos/bitacora.md). Hay que recalibrarlos
corriendo la demo sobre el dataset real antes de
usar estos números para las conclusiones del informe.
"""

from pathlib import Path

# --- Preprocesamiento y modelo ---

K_DEFECTO: int = 10
# PROVISORIO: calibrar sobre MovieLens real. Motivo: umbral de filtro y
# dimensión latente chicos a propósito, para no descartar de más sobre
# 943 usuarios x 1682 películas y para que ALS/GD corran rápido en una demo
# en vivo (mismo valor para ambos por diseño).

ETA_DEFECTO: float = 1e-4
# PROVISORIO: calibrar sobre MovieLens real. Motivo: paso chico a propósito
# — sobre la matriz completa los gradientes de la SCE tienen magnitud
# grande, y un eta mayor arriesga que f aumente en vez de bajar
# (paso 5 del algoritmo).

EPSILON_DEFECTO: float = 1e-2
# PROVISORIO: calibrar sobre MovieLens real. Motivo: tolerancia laxa a
# propósito — sobre el dataset completo la SCE arranca en el orden de los
# miles, así que una tolerancia mucho más chica podría no cortar nunca
# antes de max_iter.

MAX_ITER_DEFECTO: int = 1000
# PROVISORIO: calibrar sobre MovieLens real. Motivo: cota para que la demo
# en vivo no corra indefinidamente si epsilon no se alcanza.

SEMILLA_DEFECTO: int = 42
# PROVISORIO: calibrar sobre MovieLens real. Motivo: semilla arbitraria
# fija, solo para que la demo sea reproducible entre corridas; no
# hay un criterio de "mejor" semilla.

ESCALA_INICIALIZACION_DEFECTO: float = 1.0
# PROVISORIO: calibrar sobre MovieLens real. Motivo: escala simple (U0, V0
# uniformes en [0, 1)), del mismo orden que las calificaciones más chicas
# (1 a 5), sin sobreestimar r_hat desde el arranque.

# --- Demo ---

TOP_N_DEFECTO: int = 10
# PROVISORIO: calibrar sobre MovieLens real. Motivo: lo que entra cómodo en pantalla durante la demo; a diferencia
# de los anteriores, no depende de la escala del dataset.

USUARIO_DEFECTO: int = 1
# Id crudo de usuario de MovieLens (no índice reindexado): el primer id del
# archivo. Si el --k pedido lo filtra, main() lanza UsuarioNoEncontradoError
# con un mensaje claro en vez de fallar con un error críptico de índice.

RUTA_DATOS_DEFECTO: Path = Path("data/ml-100k")
# Carpeta con `u.data` y `u.item`, la misma que usa scripts/descargar_movielens.py.

RUTA_GRAFICO_DEFECTO: Path = Path("convergencia.png")
# Dónde se guarda el gráfico de convergencia; no hay un criterio
# más allá de que quede al lado de donde se corrió la demo.
