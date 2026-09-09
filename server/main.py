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
from server.api import datasets, infer, models, system, training
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
    app.include_router(system.router)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    if paths.STATIC_DIR.is_dir():
        # HTML 入口禁缓存（no-cache：可缓存但必须带条件请求重新验证）：前端每次构建
        # 产物哈希都会变，若浏览器启发式缓存了 index.html，会继续引用已被新构建清掉的
        # 旧哈希 JS，出现「改了没生效」的假象；带哈希的 assets 长缓存没有问题
        class _NoCacheHTML(StaticFiles):
            def file_response(self, *args, **kwargs):
                response = super().file_response(*args, **kwargs)
                if response.media_type == "text/html":
                    response.headers["cache-control"] = "no-cache"
                return response

        app.mount("/", _NoCacheHTML(directory=paths.STATIC_DIR, html=True), name="static")
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
    # 0.0.0.0：与老版 webui（gradio server_name="0.0.0.0"）对齐，允许局域网其他设备访问。
    # 该服务无任何鉴权，绑定全网卡意味着局域网内任何人都能上传/删除数据集与模型、发起训练
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
