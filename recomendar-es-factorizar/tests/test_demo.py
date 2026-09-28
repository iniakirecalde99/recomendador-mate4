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
    (directorio / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n"
        "1,1,5.0,100\n"
        "1,2,4.0,101\n"
        "2,1,3.0,102\n"
        "2,2,2.5,103\n"
        "2,3,4.0,104\n"
        "3,2,5.0,105\n"
        "3,3,1.0,106\n"
        "3,4,3.5,107\n"
        "4,1,4.0,108\n"
        "4,3,2.0,109\n"
        "4,4,5.0,110\n",
        encoding="utf-8",
    )
    (directorio / "movies.csv").write_text(
        "movieId,title,genres\n"
        "1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy\n"
        "2,GoldenEye (1995),Action|Adventure|Thriller\n"
        "3,Nixon (1995),Drama\n"
        '4,"Copycat, The (1995)",Crime|Drama|Horror|Mystery|Thriller\n',
        encoding="utf-8",
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
