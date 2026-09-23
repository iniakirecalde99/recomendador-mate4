"""Excepciones propias del proyecto, con mensajes que dicen qué falló y dónde."""


class ErrorFactorizacion(Exception):
    """Excepción base del paquete; nunca se instancia directamente."""


class ErrorDescargaDataset(ErrorFactorizacion):
    """La descarga de MovieLens falló (por ejemplo, sin red).

    Args:
        url: la URL que se intentó descargar y desde donde se puede bajar a mano.
        causa: la excepción original que disparó la falla, si la hay.
    """

    def __init__(self, url: str, causa: Exception | None = None) -> None:
        mensaje = (
            f"no se pudo descargar MovieLens desde {url}. "
            "Descargalo a mano desde esa URL y descomprimilo en data/."
        )
        super().__init__(mensaje)
        self.url = url
        self.causa = causa
