"""Configuración compartida de pytest para la demo."""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]  # habilita el fixture pytester para tests/test_conftest.py

RUTA_MOVIELENS = Path(__file__).resolve().parent.parent / "data" / "ml-100k"


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
