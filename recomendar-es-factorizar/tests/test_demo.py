"""CLI y orquestación de la demo."""

import pytest

from src import config, demo
from src.errores import UsuarioNoEncontradoError


def test_construir_parser_expone_los_argumentos_esperados_y_defaults():
    parser = demo.construir_parser()
    args = parser.parse_args([])

    assert not hasattr(args, "metodo")  # sin --metodo: la demo siempre corre ambos métodos
    assert args.umbral == config.UMBRAL_DEFECTO
    assert args.k == config.K_DEFECTO
    assert args.eta == config.ETA_DEFECTO
    assert args.epsilon == config.EPSILON_DEFECTO
    assert args.max_iter == config.MAX_ITER_DEFECTO
    assert args.semilla == config.SEMILLA_DEFECTO
    assert args.usuario == config.USUARIO_DEFECTO
    assert args.top_n == config.TOP_N_DEFECTO
    assert args.datos == config.RUTA_DATOS_DEFECTO
    assert args.grafico == config.RUTA_GRAFICO_DEFECTO
    assert not hasattr(args, "rmse")  # sin --rmse


def test_traducir_usuario_devuelve_el_indice_del_id_crudo():
    id_usuario_a_indice = {5: 0, 9: 1, 12: 2}

    assert demo._traducir_usuario(id_usuario_a_indice, 9) == 1


def test_traducir_usuario_lanza_error_claro_si_el_usuario_fue_filtrado():
    id_usuario_a_indice = {5: 0, 9: 1}

    with pytest.raises(UsuarioNoEncontradoError) as exc_info:
        demo._traducir_usuario(id_usuario_a_indice, 42)

    assert exc_info.value.usuario_id == 42


def test_formatear_top_n_lado_a_lado_incluye_ambos_metodos():
    recomendaciones_als = [("Toy Story (1995)", 4.5), ("GoldenEye (1995)", 3.2)]
    recomendaciones_gd = [("Nixon (1995)", 4.1)]

    texto = demo._formatear_top_n_lado_a_lado(recomendaciones_als, recomendaciones_gd)

    assert "ALS" in texto
    assert "GD" in texto
    assert "Toy Story (1995)" in texto
    assert "Nixon (1995)" in texto


def _escribir_dataset_chico(directorio):
    # 4 usuarios x 4 películas, k=2: cada fila y columna tiene >= 2
    # calificaciones (no hace falta cascada de filtro). El usuario crudo 1
    # (índice 0) no calificó las películas 3 y 4, para que haya algo que
    # recomendar.
    (directorio / "u.data").write_text(
        "1\t1\t5\t100\n"
        "1\t2\t4\t101\n"
        "2\t1\t3\t102\n"
        "2\t2\t2\t103\n"
        "2\t3\t4\t104\n"
        "3\t2\t5\t105\n"
        "3\t3\t1\t106\n"
        "3\t4\t3\t107\n"
        "4\t1\t4\t108\n"
        "4\t3\t2\t109\n"
        "4\t4\t5\t110\n",
        encoding="utf-8",
    )
    (directorio / "u.item").write_bytes(
        (
            "1|Toy Story (1995)|01-Jan-1995||url1|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
            "2|GoldenEye (1995)|01-Jan-1995||url2|0|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
            "3|Nixon (1995)|01-Jan-1995||url3|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
            "4|Copycat (1995)|01-Jan-1995||url4|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
        ).encode("latin-1")
    )


def test_main_corre_extremo_a_extremo_sobre_dataset_chico_sin_lanzar_excepciones(
    tmp_path, capsys
):
    _escribir_dataset_chico(tmp_path)
    ruta_grafico = tmp_path / "convergencia.png"

    demo.main(
        [
            "--datos", str(tmp_path),
            "--grafico", str(ruta_grafico),
            "--umbral", "2",
            "--k", "2",
            "--eta", "0.001",
            "--epsilon", "1e-4",
            "--max-iter", "20",
            "--semilla", "1",
            "--usuario", "1",  # id crudo del usuario que no calificó 3 y 4
            "--top-n", "2",
        ]
    )

    salida = capsys.readouterr().out
    assert "usuarios" in salida
    assert "ALS" in salida and "GD" in salida
    assert "Top-2 recomendaciones" in salida
    assert ruta_grafico.exists()


def test_main_propaga_usuarionoencontradoerror_si_el_usuario_no_existe(tmp_path):
    _escribir_dataset_chico(tmp_path)

    with pytest.raises(UsuarioNoEncontradoError):
        demo.main(
            [
                "--datos", str(tmp_path),
                "--grafico", str(tmp_path / "convergencia.png"),
                "--umbral", "2",
                "--k", "2",
                "--eta", "0.001",
                "--epsilon", "1e-4",
                "--max-iter", "20",
                "--semilla", "1",
                "--usuario", "999",  # no existe en el dataset
                "--top-n", "2",
            ]
        )


def test_main_captura_divergenciaerror_y_no_propaga_traceback(tmp_path, capsys):
    _escribir_dataset_chico(tmp_path)

    demo.main(
        [
            "--datos", str(tmp_path),
            "--grafico", str(tmp_path / "convergencia.png"),
            "--umbral", "2",
            "--k", "2",
            "--eta", "1e10",  # eta enorme: entrenar_gd tiene que divergir
            "--epsilon", "1e-4",
            "--max-iter", "1000",
            "--semilla", "1",
            "--usuario", "1",
            "--top-n", "2",
        ]
    )

    salida = capsys.readouterr().out
    assert "diverge" in salida.lower()
    assert "Traceback" not in salida
