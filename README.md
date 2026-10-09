# Recomendar es factorizar

Demo del trabajo de promoción de Matemática 4 (Facultad de Informática, UNLP).
El código implementa la factorización **R ≈ U·Vᵀ** de la matriz
usuario–película de **MovieLens latest-small** con dos métodos: **ALS**
(mínimos cuadrados alternados) y **descenso de gradiente**. Todo el álgebra
está escrita con numpy, sin librerías que factoricen o recomienden.

## En Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iniakirecalde99/recomendar-es-factorizar/blob/google-colab/entrega.ipynb)

`entrega.ipynb` tiene todo el código del modelo en sus celdas (no importa
nada de `src/`), explicado paso a paso. Se abre con el botón de arriba y se
corre con *Entorno de ejecución → Ejecutar todo*: descarga MovieLens solo,
entrena ALS y descenso de gradiente, compara la convergencia, muestra
recomendaciones y permite probar con calificaciones propias.

## Cómo correrlo en una computadora

Requiere Python 3.11 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate          # en Linux o macOS: source .venv/bin/activate
pip install -r requirements.txt

python scripts/descargar_movielens.py   # baja MovieLens a data/ (no se versiona)
pytest -q                               # tests
```

| Comando | Qué hace |
|---|---|
| `python -m src.demo` | Entrena ALS y descenso de gradiente, compara convergencia (gráfico en `salidas/convergencia.png`), muestra el top-10 de un usuario y los extremos de cada factor latente. `--help` lista los parámetros. |
| `python -m src.front` | Genera `salidas/recomendador.html`: una página estática (sin servidor) donde calificás películas y te recomienda con la V de ALS. |

Los hiperparámetros por defecto (umbral = 40, k = 2, η = 5e-4, ε = 1,
semilla 42) están en `src/config.py`, calibrados sobre MovieLens; el criterio
de cada uno está comentado ahí.

## Cómo está organizado

- `src/`: carga de datos, modelo compartido, ALS (`als.py`), descenso de
  gradiente (`gradiente.py`), recomendaciones, demo y front.
- `entrega.ipynb`: el notebook para Google Colab.
- `experimentos/`: scripts de los experimentos de concentración.
- `tests/`: tests de cada parte del código.

## Ramas

| Rama | Qué contiene |
|---|---|
| `main` | El modelo principal: k = 2, sin regularización. Incluye los experimentos de concentración de las recomendaciones (con usuarios simulados y reales, centrado y ordenamiento relativo): con k = 2 unas pocas películas acaparan los top-10. |
| `experimento/regularizacion` | Prueba de la regularización λ para usar k > 2. Resultado: elimina las estimaciones absurdas fuera del rango, pero no reduce la concentración. Cerrada sin integrar. |
| `experimento/sesgos` | Modelo con sesgos por usuario y por película, r̂ = μ + bᵢ + cⱼ + Uᵢ·Vⱼ. Ordenar por la parte personal (U·Vᵀ) reduce la concentración por debajo de los umbrales fijados de antemano. Incluye un recomendador HTML con sesgos (`python -m experimentos.front_sesgos`). |


## Datos

MovieLens latest-small, de [GroupLens](https://grouplens.org/datasets/movielens/).
La licencia no permite redistribuirlo, así que `data/` está en `.gitignore`:
se descarga con `scripts/descargar_movielens.py`.
