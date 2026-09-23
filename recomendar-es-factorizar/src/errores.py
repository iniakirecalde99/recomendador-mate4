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


class SistemaSingularError(ErrorFactorizacion):
    """Las ecuaciones normales de una fila de ALS resultaron singulares.

    Args:
        fila: índice de la fila que se intentaba resolver.
        n_observados: cantidad de columnas observadas usadas para armar el
            sistema (típicamente insuficiente o colineal para el k pedido).
    """

    def __init__(self, fila: int, n_observados: int) -> None:
        mensaje = (
            f"las ecuaciones normales de la fila {fila} son singulares: "
            f"solo hay {n_observados} observaciones para resolver el sistema."
        )
        super().__init__(mensaje)
        self.fila = fila
        self.n_observados = n_observados


class DivergenciaError(ErrorFactorizacion):
    """El descenso de gradiente divergió: f dejó de ser finito (inf o NaN).

    Args:
        iteracion: número de iteración en la que se detectó (1-indexada).
        eta: la tasa de aprendizaje usada, probablemente demasiado grande.
    """

    def __init__(self, iteracion: int, eta: float) -> None:
        mensaje = (
            f"el descenso de gradiente diverge en la iteración {iteracion}: "
            f"f dejó de ser finito (inf o NaN) con eta={eta:.4g}; probá con un eta más chico."
        )
        super().__init__(mensaje)
        self.iteracion = iteracion
        self.eta = eta


class UsuarioNoEncontradoError(ErrorFactorizacion):
    """El usuario pedido (id crudo de MovieLens) no está en el conjunto filtrado.

    Args:
        usuario_id: el id crudo de usuario que se pidió (por ejemplo, con
            `--usuario` en la demo).
    """

    def __init__(self, usuario_id: int) -> None:
        mensaje = (
            f"el usuario {usuario_id} no está en el conjunto filtrado (no existe "
            "en MovieLens, o el filtro por --umbral lo eliminó): probá con otro "
            "--usuario, o con un --umbral más chico."
        )
        super().__init__(mensaje)
        self.usuario_id = usuario_id


class UmbralInsuficienteError(ErrorFactorizacion):
    """El umbral del filtro es menor que k.

    Tener al menos k observaciones por fila es necesario para que el
    sistema k×k de ALS tenga solución única, pero no alcanza para que esa
    solución sea razonable (la calibración con MovieLens real mostró
    `SistemaSingularError` y predicciones fuera de Ω con magnitud absurda
    incluso con umbral == k). Por eso `filtrar_por_minimo` exige
    umbral >= k.

    Args:
        umbral: el umbral de calificaciones mínimas pedido para el filtro.
        k: la dimensión latente pedida.
    """

    def __init__(self, umbral: int, k: int) -> None:
        mensaje = (
            f"el umbral del filtro ({umbral}) es menor que k ({k}): ALS necesita "
            f"al menos k={k} observaciones por fila para que el sistema k×k tenga "
            "solución única. Probá con un --umbral >= --k."
        )
        super().__init__(mensaje)
        self.umbral = umbral
        self.k = k


class ErrorDatosInsuficientes(ErrorFactorizacion):
    """El filtro por umbral mínimo dejó una matriz sin filas o sin columnas.

    Args:
        umbral: el umbral mínimo de calificaciones usado en el filtro.
    """

    def __init__(self, umbral: int) -> None:
        mensaje = (
            f"el filtro por umbral mínimo={umbral} calificaciones eliminó todas "
            "las filas o todas las columnas: no queda ningún usuario o película "
            "con al menos ese umbral de calificaciones."
        )
        super().__init__(mensaje)
        self.umbral = umbral
