import socket

from fastapi.testclient import TestClient

from server.main import create_app, find_free_port


def test_health():
    client = TestClient(create_app())
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_find_free_port_skips_occupied():
    # 起点也用系统分配的空闲端口：7861 会被真实运行中的 WebUI 占用（验收/走查时服务常驻）
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        base = probe.getsockname()[1]
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", base))
        blocker.listen(1)
        assert find_free_port(start=base, tries=10) == base + 1


def test_find_free_port_returns_none_when_all_occupied():
    # 借系统分配的空闲端口作起点，避免与机器上真实运行的服务冲突（如旧 webui 占用的 7865）
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        base = probe.getsockname()[1]
    blockers = []
    try:
        for port in range(base, base + 10):
            s = socket.socket()
            s.bind(("127.0.0.1", port))
            s.listen(1)
            blockers.append(s)
        assert find_free_port(start=base, tries=10) is None
    finally:
        for s in blockers:
            s.close()
