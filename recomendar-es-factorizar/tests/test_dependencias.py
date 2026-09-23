"""Ningún archivo de src/ importa librerías que
factoricen/recomienden (Surprise, implicit, scikit-learn) ni usa
np.linalg.inv, np.linalg.lstsq o np.linalg.pinv.

No importa src/ para ejecutar código: recorre los archivos .py con `ast`,
así el test sigue siendo válido aunque src/ todavía no exista o falle en
importar.
"""

import ast
from pathlib import Path

import pytest

RUTA_SRC = Path(__file__).resolve().parent.parent / "src"

LIBRERIAS_PROHIBIDAS = {"surprise", "implicit", "sklearn"}
FUNCIONES_PROHIBIDAS = {"inv", "lstsq", "pinv"}


def _archivos_python_en_src() -> list[Path]:
    if not RUTA_SRC.exists():
        return []
    return sorted(RUTA_SRC.rglob("*.py"))


def _parsear(archivo: Path) -> ast.Module:
    return ast.parse(archivo.read_text(encoding="utf-8"), filename=str(archivo))


@pytest.fixture
def archivos_src() -> list[Path]:
    """Archivos .py de src/; si no hay ninguno, se saltea con motivo (nada
    para verificar todavía, no es lo mismo que "no hay violaciones")."""
    archivos = _archivos_python_en_src()
    if not archivos:
        pytest.skip(
            f"no hay archivos .py en {RUTA_SRC} todavía: nada que verificar"
        )
    return archivos


def test_src_no_importa_librerias_prohibidas(archivos_src: list[Path]):
    violaciones = []
    for archivo in archivos_src:
        arbol = _parsear(archivo)
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                for alias in nodo.names:
                    raiz = alias.name.split(".")[0]
                    if raiz in LIBRERIAS_PROHIBIDAS:
                        violaciones.append(f"{archivo}:{nodo.lineno}: import {alias.name}")
            elif isinstance(nodo, ast.ImportFrom):
                raiz = (nodo.module or "").split(".")[0]
                if raiz in LIBRERIAS_PROHIBIDAS:
                    violaciones.append(f"{archivo}:{nodo.lineno}: from {nodo.module} import ...")

    assert not violaciones, (
        "Import prohibido (Surprise/implicit/sklearn) en:\n" + "\n".join(violaciones)
    )


def test_src_no_usa_inv_lstsq_pinv(archivos_src: list[Path]):
    violaciones = []
    for archivo in archivos_src:
        arbol = _parsear(archivo)
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Attribute) and nodo.attr in FUNCIONES_PROHIBIDAS:
                violaciones.append(f"{archivo}:{nodo.lineno}: uso de .{nodo.attr}")
            elif isinstance(nodo, ast.ImportFrom) and (nodo.module or "").split(".")[-1] == "linalg":
                for alias in nodo.names:
                    if alias.name in FUNCIONES_PROHIBIDAS:
                        violaciones.append(
                            f"{archivo}:{nodo.lineno}: from {nodo.module} import {alias.name}"
                        )

    assert not violaciones, (
        "Uso prohibido de np.linalg.inv/lstsq/pinv en:\n" + "\n".join(violaciones)
    )
