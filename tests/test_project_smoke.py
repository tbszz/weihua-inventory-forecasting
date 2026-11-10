from pathlib import Path


def test_project_skeleton_exists():
    root = Path(__file__).parents[1]
    assert (root / "src").is_dir()
    assert (root / "data" / "raw").is_dir()
    assert (root / "reports").is_dir()

