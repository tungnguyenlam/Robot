from egograph.config import load_config


def test_base_merge_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("data:\n  clip_len: 16\n  fps: 15\nseed: 1\n")
    (tmp_path / "exp.yaml").write_text("_base_: base.yaml\ndata:\n  fps: 30\n")
    cfg = load_config(tmp_path / "exp.yaml", ["data.clip_len=8", "train.lr=1e-4"])
    assert cfg == {"data": {"clip_len": 8, "fps": 30}, "seed": 1, "train": {"lr": 1e-4}}


def test_repo_configs_load():
    from egograph.paths import REPO_ROOT
    for p in (REPO_ROOT / "configs" / "experiment").glob("*.yaml"):
        assert "data" in load_config(p)
