from scripts.launch_dashboard import normalized_env


def test_normalized_env_keeps_one_path_key():
    env = normalized_env({"PATH": "short", "Path": "longer-path", "OTHER": "value"})
    path_keys = [key for key in env if key.lower() == "path"]

    assert path_keys == ["Path"]
    assert env["Path"] == "longer-path"
    assert env["OTHER"] == "value"
