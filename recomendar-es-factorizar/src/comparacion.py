"""Comparación de ALS y descenso de gradiente."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # nunca interactivo: la demo guarda a archivo, no muestra ventanas.

import matplotlib.pyplot as plt

from src.modelo import ResultadoEntrenamiento


@dataclass
class FilaComparacion:
    """Una fila de la tabla de comparación."""

    metodo: str
    n_iteraciones: int
    motivo_corte: str
    tiempo_segundos: float
    sce_final: float


def comparar_metodos(
    resultado_als: ResultadoEntrenamiento, resultado_gd: ResultadoEntrenamiento
) -> list[FilaComparacion]:
    """Arma una fila por método con iteraciones, motivo de corte, tiempo y SCE final.

    Sin RMSE y sin partición entrenamiento/prueba: la
    SCE final es el último valor de `historial_f`, sobre Ω del conjunto
    completo filtrado.
    """
    return [
        FilaComparacion(
            metodo="ALS",
            n_iteraciones=resultado_als.n_iteraciones,
            motivo_corte=resultado_als.motivo_corte,
            tiempo_segundos=resultado_als.tiempo_segundos,
            sce_final=resultado_als.historial_f[-1],
        ),
        FilaComparacion(
            metodo="GD",
            n_iteraciones=resultado_gd.n_iteraciones,
            motivo_corte=resultado_gd.motivo_corte,
            tiempo_segundos=resultado_gd.tiempo_segundos,
            sce_final=resultado_gd.historial_f[-1],
        ),
    ]


def formatear_tabla_comparacion(filas: list[FilaComparacion]) -> str:
    """Arma la tabla de la sección 9 como texto plano alineado con f-strings.

    Sin dependencias nuevas (nada de pandas ni tabulate). Columnas: método, iteraciones, motivo de corte, tiempo (s),
    SCE final. Decimales con punto; el pasaje a coma para pegar en el
    informe es manual, fuera del código.
    """
    encabezado = (
        f"{'método':<10}{'iteraciones':>12}{'motivo de corte':>18}"
        f"{'tiempo (s)':>12}{'SCE final':>14}"
    )
    lineas = [encabezado]
    for fila in filas:
        lineas.append(
            f"{fila.metodo:<10}{fila.n_iteraciones:>12}{fila.motivo_corte:>18}"
            f"{fila.tiempo_segundos:>12.4f}{fila.sce_final:>14.4f}"
        )
    return "\n".join(lineas)


def graficar_convergencia(
    resultado_als: ResultadoEntrenamiento,
    resultado_gd: ResultadoEntrenamiento,
    ruta_salida: Path | None,
) -> None:
    """Grafica f (SCE) vs. iteración para ALS y GD en el mismo eje, escala log en y.

    Si `ruta_salida` no es None, guarda la figura ahí. Nunca llama a
    `plt.show()`: el backend es "Agg" (no interactivo), pensado para
    guardar a archivo sin bloquear la ejecución de la demo.
    """
    fig, ax = plt.subplots()
    ax.plot(range(len(resultado_als.historial_f)), resultado_als.historial_f, label="ALS")
    ax.plot(range(len(resultado_gd.historial_f)), resultado_gd.historial_f, label="GD")
    ax.set_yscale("log")
    ax.set_xlabel("iteración")
    ax.set_ylabel("SCE")
    ax.legend()

    if ruta_salida is not None:
        fig.savefig(ruta_salida)
    plt.close(fig)
