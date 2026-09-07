"""新版 WebUI 编排层入口。只做编排，算法一律复用 infer/ 与 train/。"""
import os
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server import paths
from server.api import datasets, infer, models, training
from server.tasks import task_manager


@asynccontextmanager
async def lifespan(_: FastAPI):
    """shutdown 阶段必须终止仍在运行的任务并等工作线程收尾：任务线程是 daemon 线程，
    不调 dispose 的话服务退出时线程被硬杀，其训练子进程会孤儿化（继续占 GPU/CPU 且
    无人能终止）。dispose 默认超时 10s，覆盖 kill_process_tree 自身约 6s 的阻塞；
    放到线程池里执行，避免阻塞事件循环的关闭序列。"""
    yield
    import anyio

    await anyio.to_thread.run_sync(task_manager.dispose)


def create_app() -> FastAPI:
    app = FastAPI(title="RVC WebUI", version="2.0.0-p1", lifespan=lifespan)
    app.include_router(models.router)
    app.include_router(datasets.router)
    app.include_router(infer.router)
    app.include_router(training.router)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    if paths.STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=paths.STATIC_DIR, html=True), name="static")
    return app


def find_free_port(start: int = 7861, tries: int = 10) -> int | None:
    """从 start 起向后探测 tries 个端口，返回第一个空闲端口；全部被占用则返回 None。"""
    port = start
    for _ in range(tries):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return None


def main() -> None:
    # 进程启动时设默认值（与 infer/cli.py 的惯例一致），工厂函数 create_app 保持无副作用；
    # create_app() 直起的进程（E2E 的 uvicorn 工厂调用）由 server.api.infer.get_vc_cached
    # 在首次构造 VC 时补齐同样的默认值
    os.environ.setdefault("weight_root", str(paths.WEIGHTS_DIR))
    os.environ.setdefault("index_root", str(paths.INDICES_DIR))
    os.environ.setdefault("outside_index_root", str(paths.INDICES_DIR))

    import threading
    import webbrowser

    import uvicorn

    port = find_free_port()
    if port is None:
        print("错误：7861~7870 端口均被占用，无法启动 WebUI 服务。", file=sys.stderr)
        sys.exit(1)

    threading.Timer(1.0, webbrowser.open, args=(f"http://127.0.0.1:{port}",)).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
