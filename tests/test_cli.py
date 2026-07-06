import pytest

from liteinfer.cli import main


@pytest.mark.parametrize(
    "argv",
    [
        ["quant", "--rows", "16", "--cols", "32"],
        ["kv", "--length", "10"],
        ["spec", "--tokens", "12", "--gamma", "3"],
        ["bench", "--max-gamma", "3", "--tokens", "16"],
    ],
)
def test_subcommands_exit_zero(argv, capsys):
    rc = main(argv)
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip()  # produced some output


def test_no_command_prints_help(capsys):
    rc = main([])
    assert rc == 1
    assert "usage" in capsys.readouterr().out.lower()
