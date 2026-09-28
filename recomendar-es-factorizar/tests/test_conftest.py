"""Verifica con pytest.pytester el mecanismo de skip
del marker `movielens` implementado en tests/conftest.py.

Copia el CONTENIDO REAL de tests/conftest.py (no una reescritura de su
lógica) dentro de un proyecto pytest aislado y descartable, respetando la
misma profundidad de carpetas (<sandbox>/tests/conftest.py), porque el hook
real calcula RUTA_MOVIELENS con `parent.parent`. Así, si alguien rompe el
hook real, este test lee esa rotura y falla.
"""

from pathlib import Path

RUTA_CONFTEST_REAL = Path(__file__).resolve().parent / "conftest.py"

TEST_MARCADO_MOVIELENS = '''
import pytest


@pytest.mark.movielens
def test_necesita_movielens():
    assert True
'''


def _armar_proyecto_con_el_conftest_real(pytester):
    pytester.makeini("[pytest]\nmarkers =\n    movielens: necesita MovieLens real\n")
    directorio_tests = pytester.mkdir("tests")
    contenido_real = RUTA_CONFTEST_REAL.read_text(encoding="utf-8")
    (directorio_tests / "conftest.py").write_text(contenido_real, encoding="utf-8")
    (directorio_tests / "test_ejemplo.py").write_text(TEST_MARCADO_MOVIELENS, encoding="utf-8")


def test_se_saltea_con_el_motivo_esperado_si_falta_el_dataset(pytester):
    _armar_proyecto_con_el_conftest_real(pytester)

    resultado = pytester.runpytest("-rs")

    resultado.assert_outcomes(skipped=1)
    resultado.stdout.fnmatch_lines(["*requiere el dataset real de MovieLens*"])


def test_corre_si_el_dataset_existe(pytester):
    _armar_proyecto_con_el_conftest_real(pytester)
    (pytester.path / "data" / "ml-latest-small").mkdir(parents=True)

    resultado = pytester.runpytest()

    resultado.assert_outcomes(passed=1)
