from data_sources import runtime_env


def test_remove_dead_proxy_env_only_removes_loopback_port_9(monkeypatch):
    for key in runtime_env.PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:9")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.example.com:8080")

    removed = runtime_env.remove_dead_proxy_env()

    assert set(removed) == {"HTTP_PROXY", "HTTPS_PROXY"}
    assert runtime_env.os.getenv("HTTP_PROXY") is None
    assert runtime_env.os.getenv("HTTPS_PROXY") is None
    assert runtime_env.os.getenv("ALL_PROXY") == "http://proxy.example.com:8080"
