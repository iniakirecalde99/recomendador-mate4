"""Script de descarga de MovieLens latest-small."""

import argparse
import logging
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

from src.errores import ErrorDescargaDataset

LOGGER = logging.getLogger(__name__)

URL_MOVIELENS = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
NOMBRE_CARPETA_DATASET = "ml-latest-small"


def descargar_movielens(directorio_destino: Path, url: str = URL_MOVIELENS) -> Path:
    """Descarga MovieLens latest-small a `directorio_destino` y lo descomprime.

    Si `directorio_destino/ml-latest-small/` ya existe, no vuelve a descargar. Si la
    descarga falla (por ejemplo sin red), lanza `ErrorDescargaDataset` con la
    URL para bajarlo a mano.
    """
    directorio_destino = Path(directorio_destino)
    directorio_dataset = directorio_destino / NOMBRE_CARPETA_DATASET

    if directorio_dataset.exists():
        LOGGER.info("MovieLens ya existe en %s, no se descarga de nuevo", directorio_dataset)
        return directorio_dataset

    directorio_destino.mkdir(parents=True, exist_ok=True)
    ruta_zip = directorio_destino / f"{NOMBRE_CARPETA_DATASET}.zip"

    LOGGER.info("descargando MovieLens desde %s", url)
    try:
        urlretrieve(url, ruta_zip)
    except (URLError, OSError) as excepcion:
        raise ErrorDescargaDataset(url, causa=excepcion) from excepcion

    LOGGER.info("descomprimiendo %s en %s", ruta_zip, directorio_destino)
    with zipfile.ZipFile(ruta_zip) as archivo_zip:
        archivo_zip.extractall(directorio_destino)
    ruta_zip.unlink()

    return directorio_dataset


def construir_parser() -> argparse.ArgumentParser:
    """CLI mínima: solo permite cambiar el directorio de destino (default `data/`)."""
    parser = argparse.ArgumentParser(description="Descarga y descomprime MovieLens latest-small.")
    parser.add_argument("--destino", type=Path, default=Path("data"))
    return parser


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada de `python -m scripts.descargar_movielens`."""
    logging.basicConfig(level=logging.INFO)
    args = construir_parser().parse_args(argv)
    descargar_movielens(args.destino)


if __name__ == "__main__":
    main()
