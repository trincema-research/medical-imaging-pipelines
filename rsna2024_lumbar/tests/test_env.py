import os
from pathlib import Path

import pytest

from common.utils.env import describe_kaggle_auth, find_repo_root, load_dotenv, load_repo_env, parse_dotenv
from common.utils.kaggle import ensure_kaggle_token


def test_parse_dotenv_skips_comments_and_strips_quotes(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# comment\n"
        "export KAGGLE_API_TOKEN='tok_one'\n"
        "OTHER=\"quoted\"\n"
        "BARE=plain\n"
        "\n",
        encoding="utf-8",
    )
    parsed = parse_dotenv(path)
    assert parsed["KAGGLE_API_TOKEN"] == "tok_one"
    assert parsed["OTHER"] == "quoted"
    assert parsed["BARE"] == "plain"


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("KAGGLE_API_TOKEN=from_file\n", encoding="utf-8")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "from_shell")
    load_dotenv(path, override=False)
    assert os.environ["KAGGLE_API_TOKEN"] == "from_shell"
    load_dotenv(path, override=True)
    assert os.environ["KAGGLE_API_TOKEN"] == "from_file"


def test_load_repo_env_reads_root_dotenv(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / ".env").write_text("KAGGLE_API_TOKEN=repo_tok\n", encoding="utf-8")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    found = load_repo_env(start=tmp_path, override=True)
    assert found == tmp_path / ".env"
    assert os.environ["KAGGLE_API_TOKEN"] == "repo_tok"
    assert find_repo_root(tmp_path) == tmp_path


def test_describe_kaggle_auth_never_returns_token(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_secret_should_not_appear")
    info = describe_kaggle_auth()
    assert info["token_set"] is True
    assert info["token_prefix"] == "KGAT_"
    dumped = repr(info)
    assert "secret_should_not_appear" not in dumped
    assert "KGAT_secret" not in dumped


def test_ensure_kaggle_token_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr("common.utils.kaggle.load_repo_env", lambda **k: None)
    monkeypatch.setattr("common.utils.kaggle.Path.home", lambda: tmp_path)
    with pytest.raises(SystemExit, match="KAGGLE_API_TOKEN"):
        ensure_kaggle_token()
