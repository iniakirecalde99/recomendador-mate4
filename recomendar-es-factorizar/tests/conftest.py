"""Configuración compartida de pytest para la demo."""

from pathlib import Path

import numpy as np
import pytest

pytest_plugins = ["pytester"]  # habilita el fixture pytester para tests/test_conftest.py

RUTA_MOVIELENS = Path(__file__).resolve().parent.parent / "data" / "ml-100k"

SEMILLA_FIJA = 42


@pytest.fixture
def generador_fijo() -> np.random.Generator:
    """Generator reproducible (semilla fija) para tests que no verifican
    reproducibilidad en sí (para eso, instanciar dos generadores propios con
    la misma semilla; reusar este fixture dos veces no sirve, ya que un
    Generator es stateful)."""
    return np.random.default_rng(SEMILLA_FIJA)


@pytest.fixture
def matriz_ejemplo_informe() -> tuple[np.ndarray, np.ndarray]:
    """Matriz chica de ejemplo Ana/Bruno/Carla/Diego, con k=2.

    Columnas: Toy Story, Star Wars, Fargo, Titanic, Scream.
    """
    R = np.array(
        [
            [5.0, 4.0, np.nan, 1.0, np.nan],  # Ana
            [np.nan, 5.0, 3.0, np.nan, 1.0],  # Bruno
            [1.0, np.nan, 3.0, 5.0, np.nan],  # Carla
            [np.nan, 1.0, np.nan, 4.0, 5.0],  # Diego
        ]
    )
    M = ~np.isnan(R)
    return R, M


@pytest.fixture
def matriz_pequena_aleatoria() -> tuple[np.ndarray, np.ndarray]:
    """Matriz R, M chica y aleatoria (semilla fija) para el chequeo de
    gradiente por diferencias finitas."""
    generador = np.random.default_rng(7)
    m, n = 4, 3
    R = generador.uniform(1.0, 5.0, size=(m, n))
    M = generador.uniform(size=(m, n)) < 0.7
    R[~M] = np.nan
    return R, M


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Saltea con motivo los tests marcados `movielens` si falta el dataset real."""
    if RUTA_MOVIELENS.exists():
        return
    motivo = (
        f"requiere el dataset real de MovieLens en {RUTA_MOVIELENS} "
        "(correr scripts/descargar_movielens.py)"
    )
    marca_skip = pytest.mark.skip(reason=motivo)
    for item in items:
        if "movielens" in item.keywords:
            item.add_marker(marca_skip)
