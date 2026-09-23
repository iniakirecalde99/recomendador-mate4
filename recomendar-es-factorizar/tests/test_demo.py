"""CLI y orquestación de la demo."""

import pytest

from src import config, demo


def test_construir_parser_expone_los_argumentos_esperados_y_defaults():
    parser = demo.construir_parser()
    args = parser.parse_args([])

    assert args.metodo == "ambos"
    assert args.k == config.K_DEFECTO
    assert args.eta == config.ETA_DEFECTO
    assert args.epsilon == config.EPSILON_DEFECTO
    assert args.max_iter == config.MAX_ITER_DEFECTO
    assert args.semilla == config.SEMILLA_DEFECTO
    assert args.top_n == config.TOP_N_DEFECTO
    assert not hasattr(args, "rmse")  # sin --rmse

    with pytest.raises(SystemExit):
        parser.parse_args(["--metodo", "invalido"])


def _escribir_dataset_chico(directorio):
    # 4 usuarios x 4 películas, k=2: cada fila y columna tiene >= 2
    # calificaciones (no hace falta cascada de filtro). El usuario 1 (índice
    # 0) no calificó las películas 3 y 4, para que haya algo que recomendar.
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
    tmp_path, monkeypatch, capsys
):
    _escribir_dataset_chico(tmp_path)
    monkeypatch.setattr(demo, "RUTA_DATOS_DEFECTO", tmp_path)
    monkeypatch.setattr(demo, "RUTA_GRAFICO_DEFECTO", tmp_path / "convergencia.png")

    demo.main(
        [
            "--metodo", "ambos",
            "--k", "2",
            "--eta", "0.001",
            "--epsilon", "1e-4",
            "--max-iter", "20",
            "--semilla", "1",
            "--usuario", "0",
            "--top-n", "2",
        ]
    )

    salida = capsys.readouterr().out
    assert "usuarios" in salida
    assert "ALS" in salida and "GD" in salida
    assert "Top-2 recomendaciones" in salida
    assert (tmp_path / "convergencia.png").exists()
