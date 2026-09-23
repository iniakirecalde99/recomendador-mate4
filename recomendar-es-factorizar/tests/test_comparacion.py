"""Tabla de comparación y gráfico de convergencia."""

import numpy as np

from src.comparacion import (
    FilaComparacion,
    comparar_metodos,
    formatear_tabla_comparacion,
    graficar_convergencia,
)
from src.modelo import ResultadoEntrenamiento


def _resultado_dummy(historial_f, n_iteraciones, tiempo_segundos, motivo_corte):
    return ResultadoEntrenamiento(
        U=np.zeros((1, 1)),
        V=np.zeros((1, 1)),
        historial_f=historial_f,
        n_iteraciones=n_iteraciones,
        tiempo_segundos=tiempo_segundos,
        motivo_corte=motivo_corte,
    )


def test_comparar_metodos_arma_una_fila_por_metodo():
    resultado_als = _resultado_dummy([10.0, 5.0, 3.0], n_iteraciones=2, tiempo_segundos=0.05, motivo_corte="tolerancia")
    resultado_gd = _resultado_dummy([10.0, 6.0, 4.0, 3.5], n_iteraciones=3, tiempo_segundos=0.08, motivo_corte="max_iter")

    filas = comparar_metodos(resultado_als, resultado_gd)

    assert len(filas) == 2
    assert filas[0] == FilaComparacion(
        metodo="ALS", n_iteraciones=2, motivo_corte="tolerancia", tiempo_segundos=0.05, sce_final=3.0
    )
    assert filas[1] == FilaComparacion(
        metodo="GD", n_iteraciones=3, motivo_corte="max_iter", tiempo_segundos=0.08, sce_final=3.5
    )


def test_formatear_tabla_comparacion_incluye_las_columnas_esperadas():
    filas = [
        FilaComparacion(metodo="ALS", n_iteraciones=2, motivo_corte="tolerancia", tiempo_segundos=0.05, sce_final=3.0),
        FilaComparacion(metodo="GD", n_iteraciones=3, motivo_corte="max_iter", tiempo_segundos=0.08, sce_final=3.5),
    ]

    tabla = formatear_tabla_comparacion(filas)

    # Encabezado con las columnas de la tabla.
    for columna in ["método", "iteraciones", "motivo de corte", "tiempo", "SCE"]:
        assert columna in tabla

    # Datos de cada fila presentes como texto plano, con punto decimal.
    assert "ALS" in tabla and "GD" in tabla
    assert "tolerancia" in tabla and "max_iter" in tabla
    assert "3.0" in tabla or "3.0000" in tabla


def test_graficar_convergencia_guarda_archivo_en_ruta_dada(tmp_path):
    resultado_als = _resultado_dummy([10.0, 5.0, 3.0], n_iteraciones=2, tiempo_segundos=0.05, motivo_corte="tolerancia")
    resultado_gd = _resultado_dummy([10.0, 6.0, 4.0, 3.5], n_iteraciones=3, tiempo_segundos=0.08, motivo_corte="max_iter")
    ruta_salida = tmp_path / "convergencia.png"

    graficar_convergencia(resultado_als, resultado_gd, ruta_salida)

    assert ruta_salida.exists()
    assert ruta_salida.stat().st_size > 0
