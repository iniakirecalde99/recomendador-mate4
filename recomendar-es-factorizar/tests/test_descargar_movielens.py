"""Script de descarga de MovieLens.

La descarga va mockeada en los dos casos: nunca se toca la red real.
"""

from unittest.mock import patch
from urllib.error import URLError

import pytest

from scripts.descargar_movielens import descargar_movielens
from src.errores import ErrorDescargaDataset

URL_DE_PRUEBA = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def test_descargar_movielens_informa_si_ya_existe(tmp_path, caplog):
    directorio_dataset = tmp_path / "ml-100k"
    directorio_dataset.mkdir()

    with patch("scripts.descargar_movielens.urlretrieve") as mock_urlretrieve:
        with caplog.at_level("INFO"):
            resultado = descargar_movielens(tmp_path, url=URL_DE_PRUEBA)

    assert resultado == directorio_dataset
    mock_urlretrieve.assert_not_called()
    assert "ya existe" in caplog.text


def test_descargar_movielens_falla_con_mensaje_y_url_si_no_hay_red(tmp_path):
    with patch(
        "scripts.descargar_movielens.urlretrieve",
        side_effect=URLError("sin red"),
    ):
        with pytest.raises(ErrorDescargaDataset) as info_excepcion:
            descargar_movielens(tmp_path, url=URL_DE_PRUEBA)

    assert URL_DE_PRUEBA in str(info_excepcion.value)
