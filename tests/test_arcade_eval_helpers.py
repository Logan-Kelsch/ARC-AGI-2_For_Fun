import pytest

pytest.importorskip("arc_agi")
pytest.importorskip("arcengine")

from arc_fun.arcade_eval import short_game_id


def test_short_game_id_strips_version_suffix():
    assert short_game_id("ls20-9607627b") == "ls20"


def test_short_game_id_leaves_short_id_alone():
    assert short_game_id("ls20") == "ls20"


def test_make_arcade_uses_repo_root_cache_paths(monkeypatch):
    import arc_fun.arcade_eval as arcade_eval

    captured = {}

    class FakeArcade:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(arcade_eval.arc_agi, "Arcade", FakeArcade)
    arcade_eval.make_arcade("offline")

    assert captured["environments_dir"] == str(
        arcade_eval.REPO_ROOT / "environment_files"
    )
    assert captured["recordings_dir"] == str(
        arcade_eval.REPO_ROOT / "recordings"
    )
