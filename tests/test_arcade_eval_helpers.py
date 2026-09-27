import pytest

pytest.importorskip("arc_agi")
pytest.importorskip("arcengine")

from arc_fun.arcade_eval import short_game_id


def test_short_game_id_strips_version_suffix():
    assert short_game_id("ls20-9607627b") == "ls20"


def test_short_game_id_leaves_short_id_alone():
    assert short_game_id("ls20") == "ls20"
