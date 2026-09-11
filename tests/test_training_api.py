"""训练 API（4 步 + 一键 + 任务查询/取消）路由测试。

默认用 StubTaskManager 记录 create_task 的入参（不跑子进程），断言各步骤的命令拼装、
日志文件与参数校验；只有「fit 前置失败/成功 → 任务终态」用例换成真 TaskManager + 假
Popen + tmp_path 造产物，验证编排层 setup 钩子的完整链路（硬性要求③⑧）。

autouse 夹具把 paths.ROOT / paths.LOGS_DIR 指到 tmp_path：命令串里的仓库根、底模存在性
检查与日志落盘都不触碰真实 assets/ 与 logs/（本机 assets/pretrained_v2 有真实底模，
不隔离会让 -pg/-pd 断言随机器状态漂移）。
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import commands, paths
from server.api import training
from server.main import create_app
from server.tasks import TERMINAL_STATES, TaskManager

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
CPU = os.cpu_count()


# ---------------------------------------------------------------------------
# 替身与夹具
# ---------------------------------------------------------------------------


class StubTaskManager:
    """记录 create_task 入参的替身；快照由测试直接塞进 snapshots。

    内置最小队列语义：已有非终态任务时新任务登记为 pending（queue_position 在
    pending 中顺延），否则即刻 running——API 层 queued/queue_position 的响应映射
    与同名实验互斥用它验证。list_tasks 按 snapshots 的插入序输出，测试手工播种
    快照时无需再同步单独的顺序表。
    """

    def __init__(self):
        self.created = []  # create_task 入参（按创建序）
        self.order = []  # task_id 创建序（与 snapshots 插入序一致，保留供断言）
        self.snapshots = {}
        self.cancelled = []
        self.cleared = 0  # clear_queue 调用次数

    def create_task(
        self, name, cmds, log_path, truncate: bool = True, *, setup=None,
        definition=None, task_id=None, created_at=None,
    ):
        self._seq = getattr(self, "_seq", 0) + 1
        tid = task_id or "task-%d" % self._seq
        states = [snapshot["state"] for snapshot in self.snapshots.values()]
        if any(state not in TERMINAL_STATES for state in states):
            position = states.count("pending") + 1
            self.snapshots[tid] = _snapshot(
                tid, name, state="pending", queue_position=position
            )
        else:
            self.snapshots[tid] = _snapshot(tid, name)
        self.created.append(
            {
                "id": tid,
                "name": name,
                "cmds": list(cmds),
                "log_path": log_path,
                "truncate": truncate,
                "setup": setup,
                "definition": definition,
            }
        )
        self.order.append(tid)
        return tid

    def get_task(self, task_id):
        return self.snapshots.get(task_id)

    def list_tasks(self):
        return list(self.snapshots.values())

    def cancel(self, task_id):
        self.cancelled.append(task_id)
        return True

    def clear_queue(self):
        self.cleared += 1
        return {"stopped_task": None, "cancelled_pending": 0}


def _snapshot(
    task_id, name="fit", state="running", progress=None, error=None, logs=(),
    current_cmd=None, queue_position=None,
):
    return {
        "id": task_id,
        "name": name,
        "state": state,
        "progress": progress,
        "error": error,
        "cmds": ["cmd"],
        "current_cmd": current_cmd,
        "queue_position": queue_position,
        "log_path": "log-path",
        "created_at": 1.0,
        "started_at": 2.0,
        "finished_at": None,
        "log_seq": len(list(logs)),
        "logs": list(logs),
    }


@pytest.fixture(autouse=True)
def isolated_fs(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "logs")


@pytest.fixture(autouse=True)
def fake_device(monkeypatch):
    """device/is_half 的惰性解析会加载 torch（Configs 的探测在测试环境不可复现），
    统一替换为确定值；需要特定 device 的用例可在用例内再次覆盖。"""
    monkeypatch.setattr(commands, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(training, "resolve_is_half", lambda: False)


@pytest.fixture
def stub(monkeypatch):
    manager = StubTaskManager()
    monkeypatch.setattr(training, "task_manager", manager)
    monkeypatch.setattr(training, "_TASK_META", {})
    return manager


@pytest.fixture
def client():
    return TestClient(create_app())


def _touch(path, name):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_bytes(b"x")


def _fabricate_features(exp_dir, stems=("a",), *, with_16k=True):
    """造齐 fit 前置需要的产物目录（generate_filelist 与 validate_feature_outputs 的并集）。"""
    for name in ("0_gt_wavs", "1_16k_wavs", "3_feature768", "2a_f0", "2b-f0nsf"):
        (exp_dir / name).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        _touch(exp_dir / "0_gt_wavs", f"{stem}.wav")
        _touch(exp_dir / "3_feature768", f"{stem}.npy")
        _touch(exp_dir / "2a_f0", f"{stem}.wav.npy")
        _touch(exp_dir / "2b-f0nsf", f"{stem}.wav.npy")
        if with_16k:
            _touch(exp_dir / "1_16k_wavs", f"{stem}.wav")


FIT_BODY = {
    "exp_name": "mi-test",
    "sr": "48k",
    "version": "v2",
    "if_f0": True,
    "total_epoch": 20,
    "save_every_epoch": 5,
    "batch_size": 8,
}


# ---------------------------------------------------------------------------
# 1. 各步骤命令拼装
# ---------------------------------------------------------------------------


def test_preprocess_builds_command(stub, client, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    resp = client.post(
        "/api/train/preprocess",
        json={"exp_name": "mi-test", "dataset_dir": str(dataset)},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "task-1"
    assert body["queued"] is False and body["queue_position"] is None
    call = stub.created[0]
    assert call["name"] == "preprocess"
    # 请求未带 sr → 钉住默认值 40k（对齐 webui 默认与官方底模推荐）
    assert call["cmds"] == [
        commands.build_preprocess_cmd(str(dataset), 40000, CPU, "mi-test", False, 3.7)
    ]
    assert call["log_path"] == paths.LOGS_DIR / "mi-test" / "preprocess.log"
    assert call["truncate"] is True  # 每阶段启动前截断，对齐 webui
    assert call["setup"] is None


def test_preprocess_custom_sr_and_np(stub, client, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    resp = client.post(
        "/api/train/preprocess",
        json={"exp_name": "mi-test", "dataset_dir": str(dataset), "sr": "40k", "n_p": 4},
    )

    assert resp.status_code == 200
    assert commands.SR_DICT["40k"] == 40000  # sr 走 SR_DICT 转 int
    assert str(40000) in stub.created[0]["cmds"][0]
    assert " 4 " in stub.created[0]["cmds"][0]


def test_extract_runs_f0_then_hubert_in_one_task(stub, client, monkeypatch):

    resp = client.post("/api/train/extract", json={"exp_name": "mi-test"})

    assert resp.status_code == 200
    call = stub.created[0]
    assert call["name"] == "extract"
    assert call["cmds"] == [
        commands.build_extract_f0_cmd("mi-test", CPU, "rmvpe"),
        commands.build_extract_hubert_cmd("mi-test", "v2", False),
    ]  # 顺序执行：f0 完成后才起 hubert
    assert call["log_path"] == paths.LOGS_DIR / "mi-test" / "extract_f0_feature.log"
    assert call["setup"] is not None  # 启动前校验切分产物，失败 → 任务 failed


def test_extract_without_f0_single_cmd(stub, client, monkeypatch):

    resp = client.post(
        "/api/train/extract", json={"exp_name": "mi-test", "if_f0": False, "f0_method": "pm"}
    )

    assert resp.status_code == 200
    cmds = stub.created[0]["cmds"]
    assert len(cmds) == 1 and "extract_hubert_feature.py" in cmds[0]
    assert "extract_f0.py" not in cmds[0]


def test_fit_builds_command_without_pretrained(stub, client):
    """底模缺失（隔离后的 tmp ROOT 下不存在）→ 不带 -pg/-pd，并由 setup 返回提示行。"""
    resp = client.post("/api/train/fit", json=FIT_BODY)

    assert resp.status_code == 200
    call = stub.created[0]
    assert call["name"] == "fit"
    assert call["cmds"] == [
        commands.build_fit_cmd(
            "mi-test", "48k", True, 8, 20, 5, False, "", "", version="v2"
        )
    ]
    assert "-pg" not in call["cmds"][0] and "-pd" not in call["cmds"][0]
    # 硬性要求④：fit 用任务专属日志文件，不得复用 train.log（FileHandler 双写会污染解析锚点）
    assert call["log_path"] == paths.LOGS_DIR / "mi-test" / "train_task_fit.log"
    assert call["log_path"].name != "train.log"
    assert call["truncate"] is True
    assert call["setup"] is not None


def test_fit_builds_command_with_pretrained(stub, client, tmp_path):
    _touch(tmp_path / "assets" / "pretrained_v2", "f0G48k.pth")
    _touch(tmp_path / "assets" / "pretrained_v2", "f0D48k.pth")

    resp = client.post("/api/train/fit", json=FIT_BODY)

    assert resp.status_code == 200
    assert "-pg assets/pretrained_v2/f0G48k.pth" in stub.created[0]["cmds"][0]
    assert "-pd assets/pretrained_v2/f0D48k.pth" in stub.created[0]["cmds"][0]


def test_index_builds_command(stub, client):
    resp = client.post("/api/train/index", json={"exp_name": "mi-test"})

    assert resp.status_code == 200
    call = stub.created[0]
    assert call["name"] == "index"
    assert call["cmds"] == [commands.build_index_cmd("mi-test", "v2", CPU)]
    assert call["log_path"] == paths.LOGS_DIR / "mi-test" / "train_index.log"


def test_pipeline_builds_all_steps_in_one_task(stub, client, monkeypatch, tmp_path):
    """一键 = 单任务多 cmd：preprocess → precheck → f0 → hubert → fit 前置 → fit → index。
    任一 cmd 失败/取消后不再启动后续（tasks.py 已保证），两个 Python 前置以子进程 cmd
    插在对应子进程之后，时序与分步流程一致。"""
    monkeypatch.setattr(training, "resolve_is_half", lambda: False)
    monkeypatch.setattr(commands, "_resolve_device", lambda: "cpu")
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    resp = client.post(
        "/api/train/pipeline", json={**FIT_BODY, "dataset_dir": str(dataset)}
    )

    assert resp.status_code == 200
    call = stub.created[0]
    assert call["name"] == "pipeline"
    assert [c.split(" ")[1] for c in call["cmds"]] == [
        "train/preprocess.py",
        "-m",
        "train/dataset/extract_f0.py",
        "train/dataset/extract_hubert_feature.py",
        "-m",
        "train/train.py",
        "train/train_index.py",
    ]
    assert call["cmds"][1] == commands.build_precheck_cmd("mi-test")
    assert call["cmds"][4] == commands.build_fitprep_cmd("mi-test", "48k", "v2", True)
    assert call["cmds"][5] == commands.build_fit_cmd(
        "mi-test", "48k", True, 8, 20, 5, False, "", "", version="v2"
    )
    assert call["log_path"] == paths.LOGS_DIR / "mi-test" / "pipeline_task.log"
    # setup 只承担 batch_size 默认值说明（显式传值时无输出）；产物校验必须走 precheck /
    # fitprep 子进程——任务级 setup 在首个 cmd 前执行，拿去校验会误判新实验（见 fitprep 时序）
    assert call["setup"] is not None
    assert call["setup"]() == []
    # 前端进度条需要 total_epoch；加权进度需要 stages 表（与 cmds 等长）
    assert training._TASK_META["task-1"] == {
        "total_epoch": 20,
        "exp_name": "mi-test",
        "pipeline_stages": ["preprocess", "preprocess", "extract", "extract", "fit", "fit", "index"],
    }


def test_pipeline_without_f0_skips_f0_cmd(stub, client, monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    resp = client.post(
        "/api/train/pipeline",
        json={**FIT_BODY, "if_f0": False, "dataset_dir": str(dataset)},
    )

    assert resp.status_code == 200
    joined = "\n".join(stub.created[0]["cmds"])
    assert "extract_f0.py" not in joined and "extract_hubert_feature.py" in joined


def test_pipeline_fresh_experiment_starts_preprocess(client, real_manager):
    """回归：pipeline 不得用任务级 setup 做产物校验——那会在 preprocess 执行前就判死
    新实验。真管理器下，全新实验的 pipeline 必须先把第一个 cmd 跑起来。"""
    dataset = paths.LOGS_DIR.parent / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)

    resp = client.post(
        "/api/train/pipeline", json={**FIT_BODY, "dataset_dir": str(dataset)}
    )
    task_id = resp.json()["task_id"]

    wait_until(lambda: real_manager[0].get_task(task_id)["state"] == "success")
    started = real_manager[1]
    assert started[0].startswith(f'"{PY}" train/preprocess.py')
    assert len(started) == 7  # 全部 cmd 按序启动（假 Popen 不真执行）
    assert "precheck" in started[1] and "fitprep" in started[4]


# ---------------------------------------------------------------------------
# 2. 参数校验（400）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exp_name",
    ["", "../evil", "a/b", "..", ".", 'a"b', "a\\b", "a b", "a\tb", "a\nb", "a$b", "a`b", "a\0b"],
)
def test_invalid_exp_name_rejected(stub, client, tmp_path, exp_name):
    # a\0b：NUL 漏到任务编排层（mkdir/日志路径）会 ValueError，必须在 400 拦下
    # （与 server.api.datasets._check_name 同步补的判定）
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    resp = client.post(
        "/api/train/preprocess",
        json={"exp_name": exp_name, "dataset_dir": str(dataset)},
    )

    assert resp.status_code == 400
    assert stub.created == []  # 未创建任务


def test_missing_dataset_dir_rejected(stub, client, tmp_path):
    resp = client.post(
        "/api/train/preprocess",
        json={"exp_name": "mi-test", "dataset_dir": str(tmp_path / "nope")},
    )

    assert resp.status_code == 400
    assert "数据集目录不存在" in resp.json()["detail"]
    assert stub.created == []


@pytest.mark.parametrize(
    "dataset_dir", ['a"b', "a$b", "a`b", "/data/ds\\", "a\nb", "a\rb", ""]
)
def test_dataset_dir_with_shell_metacharacters_rejected(stub, client, dataset_dir):
    """命令串按双引号包裹路径：引号/换行破坏参数边界，$ 与反引号触发命令替换，
    结尾反斜杠会吞掉闭引号——都在参数校验阶段拒绝。"""
    resp = client.post(
        "/api/train/preprocess",
        json={"exp_name": "mi-test", "dataset_dir": dataset_dir},
    )

    assert resp.status_code == 400
    assert "数据集路径非法" in resp.json()["detail"]
    assert stub.created == []


@pytest.mark.parametrize(
    "endpoint, body, field",
    [
        ("/api/train/preprocess", {"dataset_dir": "x"}, "sr"),
        ("/api/train/extract", {}, "f0_method"),
        ("/api/train/extract", {}, "version"),
        ("/api/train/fit", dict(FIT_BODY), "sr"),
        ("/api/train/fit", dict(FIT_BODY), "version"),
        ("/api/train/index", {}, "version"),
    ],
)
def test_enum_fields_rejected(stub, client, tmp_path, endpoint, body, field):
    payload = dict(body)
    payload["exp_name"] = "mi-test"
    payload[field] = "bogus"
    if endpoint == "/api/train/preprocess":
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        payload["dataset_dir"] = str(dataset)

    resp = client.post(endpoint, json=payload)

    assert resp.status_code == 400
    assert stub.created == []


@pytest.mark.parametrize(
    "field", ["total_epoch", "save_every_epoch", "batch_size"]
)
def test_non_positive_numeric_fields_rejected(stub, client, field):
    resp = client.post("/api/train/fit", json={**FIT_BODY, field: 0})

    assert resp.status_code == 400
    assert field in resp.json()["detail"]
    assert stub.created == []


def test_np_must_be_positive(stub, client, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    resp = client.post(
        "/api/train/preprocess",
        json={"exp_name": "mi-test", "dataset_dir": str(dataset), "n_p": 0},
    )

    assert resp.status_code == 400
    assert stub.created == []


def test_v1_32k_normalized_to_40k_in_fit(stub, client):
    """硬性要求①：webui.py:1118-1123 change_version19 —— v1 无 32k 档，一律归一化 40k
    （train.py / config 模板 / filelist 的 mute 文件名都依赖该一致性）。"""
    resp = client.post("/api/train/fit", json={**FIT_BODY, "sr": "32k", "version": "v1"})

    assert resp.status_code == 200
    assert "-sr 40k" in stub.created[0]["cmds"][0]


def test_v1_32k_normalized_to_40k_in_pipeline(stub, client, monkeypatch, tmp_path):
    """归一化必须发生在所有校验之后、使用之前：preprocess 的采样率整数同样用 40k。"""
    monkeypatch.setattr(training, "resolve_is_half", lambda: False)
    monkeypatch.setattr(commands, "_resolve_device", lambda: "cpu")
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    resp = client.post(
        "/api/train/pipeline",
        json={**FIT_BODY, "sr": "32k", "version": "v1", "dataset_dir": str(dataset)},
    )

    assert resp.status_code == 200
    assert " 40000 " in stub.created[0]["cmds"][0]  # preprocess
    assert "-sr 40k" in stub.created[0]["cmds"][5]  # fit
    assert " 40k " in stub.created[0]["cmds"][4]  # fitprep


def test_v2_32k_keeps_32k(stub, client):
    resp = client.post("/api/train/fit", json={**FIT_BODY, "sr": "32k"})

    assert resp.status_code == 200
    assert "-sr 32k" in stub.created[0]["cmds"][0]


# ---------------------------------------------------------------------------
# 2.5 batch_size 设备默认（webui.py:175-183 显存÷2；None → 解析值 + 任务日志说明行）
# ---------------------------------------------------------------------------


def _stub_gpu_memory(monkeypatch, memory):
    """替换 GPU 探测底层（commands 里两个入口共用的唯一数据源，替换一次即可）：
    memory 是 GPU_MEMORY 的值列表，8.4 ≈ 8 GiB 卡（total/1024**3 + 0.4）。"""
    monkeypatch.setattr(commands, "_eligible_gpu_memory_gb", lambda: memory)


def _fit_body_without_batch_size():
    return {key: value for key, value in FIT_BODY.items() if key != "batch_size"}


def test_fit_without_batch_size_uses_device_default(stub, client, monkeypatch):
    """batch_size 缺省 → 命令用解析值（8 GiB 卡 ÷2 = 4），任务日志首行说明来源。"""
    _stub_gpu_memory(monkeypatch, [8.4])
    _fabricate_features(paths.LOGS_DIR / "mi-test")  # setup 会真跑 filelist 生成

    resp = client.post("/api/train/fit", json=_fit_body_without_batch_size())

    assert resp.status_code == 200
    call = stub.created[0]
    assert " -bs 4 " in call["cmds"][0]
    assert call["setup"] is not None
    lines = call["setup"]()
    assert lines[0] == "batch_size 未指定，按设备默认使用 4（可用显卡最小显存 8.4 GB ÷ 2）"


def test_fit_without_batch_size_no_gpu_falls_back_to_one(stub, client, monkeypatch):
    _stub_gpu_memory(monkeypatch, [])
    _fabricate_features(paths.LOGS_DIR / "mi-test")

    resp = client.post("/api/train/fit", json=_fit_body_without_batch_size())

    assert resp.status_code == 200
    call = stub.created[0]
    assert " -bs 1 " in call["cmds"][0]
    assert call["setup"]()[0] == "batch_size 未指定，按设备默认使用 1（无可用显卡）"


def test_fit_explicit_batch_size_bypasses_default(stub, client, monkeypatch):
    """显式值不受设备默认影响，也不产生说明行（底模提示行照旧）。"""
    _stub_gpu_memory(monkeypatch, [8.4])
    _fabricate_features(paths.LOGS_DIR / "mi-test")

    resp = client.post("/api/train/fit", json={**FIT_BODY, "batch_size": 6})

    assert resp.status_code == 200
    call = stub.created[0]
    assert " -bs 6 " in call["cmds"][0]
    assert all("batch_size 未指定" not in line for line in call["setup"]())


def test_fit_null_batch_size_equals_omitted(stub, client, monkeypatch):
    """显式 null 与省略字段同轨（前端「自动」态就是发 null/不发字段）。"""
    _stub_gpu_memory(monkeypatch, [8.4])

    resp = client.post("/api/train/fit", json={**FIT_BODY, "batch_size": None})

    assert resp.status_code == 200
    assert " -bs 4 " in stub.created[0]["cmds"][0]


def test_fit_zero_batch_size_still_rejected(stub, client, monkeypatch):
    """默认值逻辑不放松校验：显式 0 依旧 400（解析值自带 max(1, …) 下限）。"""
    _stub_gpu_memory(monkeypatch, [8.4])

    resp = client.post("/api/train/fit", json={**FIT_BODY, "batch_size": 0})

    assert resp.status_code == 400
    assert stub.created == []


def test_pipeline_without_batch_size_uses_device_default(stub, client, monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _stub_gpu_memory(monkeypatch, [8.4])

    resp = client.post(
        "/api/train/pipeline",
        json={**_fit_body_without_batch_size(), "dataset_dir": str(dataset)},
    )

    assert resp.status_code == 200
    call = stub.created[0]
    assert " -bs 4 " in call["cmds"][5]  # fit 是第 6 个 cmd
    # pipeline 的 setup 只输出说明行，不做产物校验（时序见 start_pipeline docstring）
    assert call["setup"]() == [
        "batch_size 未指定，按设备默认使用 4（可用显卡最小显存 8.4 GB ÷ 2）"
    ]


def test_train_defaults_returns_resolved_batch_size(client, monkeypatch):
    _stub_gpu_memory(monkeypatch, [8.4])

    resp = client.get("/api/train/defaults")

    assert resp.status_code == 200
    assert resp.json() == {"batch_size": 4}


def test_train_defaults_without_gpu_returns_one(client, monkeypatch):
    _stub_gpu_memory(monkeypatch, [])

    resp = client.get("/api/train/defaults")

    assert resp.status_code == 200
    assert resp.json() == {"batch_size": 1}


# ---------------------------------------------------------------------------
# 3. fit 的 setup 钩子（真跑，验证硬性要求②③⑧）
# ---------------------------------------------------------------------------


def test_fit_setup_generates_filelist_config_and_pretrained_notices(stub, client):
    exp_dir = paths.LOGS_DIR / "mi-test"
    _fabricate_features(exp_dir)

    resp = client.post("/api/train/fit", json=FIT_BODY)
    assert resp.status_code == 200

    lines = stub.created[0]["setup"]()  # 直接执行任务前置钩子

    assert (exp_dir / "filelist.txt").is_file()
    assert (exp_dir / "config.json").is_file()
    # 硬性要求②：底模缺失必须可见（写进任务日志与 SSE，而不是静默从零训练）
    assert len(lines) == 2
    assert "生成器预训练模型不存在，将不使用" in lines[0]
    assert "判别器预训练模型不存在，将不使用" in lines[1]
    assert "assets/pretrained_v2/f0G48k.pth" in lines[0]


def test_fit_setup_with_pretrained_returns_no_notice(stub, client, tmp_path):
    _touch(tmp_path / "assets" / "pretrained_v2", "f0G48k.pth")
    _touch(tmp_path / "assets" / "pretrained_v2", "f0D48k.pth")
    _fabricate_features(paths.LOGS_DIR / "mi-test")

    client.post("/api/train/fit", json=FIT_BODY)

    assert stub.created[0]["setup"]() == []


class _FakeProcess:
    def __init__(self, returncode=0, delay=0):
        self.returncode = None
        self._rc = returncode
        self._delay = delay
        self.pid = 42424

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self._delay > 0:
            self._delay -= 1
            return None
        self.returncode = self._rc
        return self.returncode


@pytest.fixture
def real_manager(monkeypatch, tmp_path):
    """真 TaskManager（假 Popen：测试场景不允许真起子进程），验证 API → setup → 终态链路。"""
    started = []

    def fake_popen(cmd, **kwargs):
        started.append(cmd)
        return _FakeProcess(0)

    manager = TaskManager(poll_interval=0.01, popen=fake_popen)
    monkeypatch.setattr(training, "task_manager", manager)
    monkeypatch.setattr(training, "_TASK_META", {})
    yield manager, started
    manager.dispose()


def wait_until(predicate, timeout=5.0, message="等待条件超时"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def test_fit_data_not_ready_becomes_failed_task_not_500(client, real_manager):
    """硬性要求③：generate_filelist 的 ValueError（数据未就绪）→ 任务 failed，非 500。"""
    client.post("/api/train/fit", json=FIT_BODY)  # exp 目录无任何产物
    task_id = real_manager[0].list_tasks()[0]["id"]

    wait_until(
        lambda: real_manager[0].get_task(task_id)["state"] == "failed"
    )

    snapshot = real_manager[0].get_task(task_id)
    assert "没有可用于训练的有效音频" in snapshot["error"]
    assert real_manager[1] == []  # 前置失败，未启动任何子进程


def test_fit_validation_failure_becomes_failed_task(client, real_manager):
    """硬性要求③：validate_feature_outputs 的 RuntimeError（校验失败）同样任务化。
    造出 filelist 能生成（0_gt/特征/f0 齐全）但 1_16k_wavs 缺失的产物树。"""
    _fabricate_features(paths.LOGS_DIR / "mi-test", with_16k=False)

    client.post("/api/train/fit", json=FIT_BODY)
    task_id = real_manager[0].list_tasks()[0]["id"]

    wait_until(
        lambda: real_manager[0].get_task(task_id)["state"] == "failed"
    )

    assert "HuBERT特征提取没有生成有效结果" in real_manager[0].get_task(task_id)["error"]
    assert real_manager[1] == []


def test_fit_success_flow_appends_pretrained_notice_to_task_log(client, real_manager):
    """端到端（无真子进程）：setup 成功 → 任务 success，底模提示行进入任务日志缓冲
    （SSE/GET 的任务日志视图），fit cmd 正常启动。"""
    _fabricate_features(paths.LOGS_DIR / "mi-test")

    resp = client.post("/api/train/fit", json=FIT_BODY)
    task_id = resp.json()["task_id"]

    wait_until(
        lambda: real_manager[0].get_task(task_id)["state"] == "success"
    )
    logs = real_manager[0].get_task(task_id)["logs"]
    assert any("生成器预训练模型不存在" in line for line in logs)
    assert len(real_manager[1]) == 1 and "train/train.py" in real_manager[1][0]


# ---------------------------------------------------------------------------
# 4. fitprep 子进程入口（pipeline 的 fit 前置 cmd）
# ---------------------------------------------------------------------------


def test_fitprep_main_generates_prep_and_prints_notices(capsys):
    exp_dir = paths.LOGS_DIR / "mi-test"
    _fabricate_features(exp_dir)

    code = training.fitprep_main(["fitprep", "mi-test", "48k", "v2", "1"])

    assert code == 0
    assert (exp_dir / "filelist.txt").is_file()
    out = capsys.readouterr().out
    assert "生成器预训练模型不存在，将不使用" in out  # 提示随 stdout 进任务日志缓冲
    assert "训练前置生成完成" in out


def test_fitprep_main_failure_returns_nonzero_and_prints_reason(capsys):
    code = training.fitprep_main(["fitprep", "empty-exp", "48k", "v2", "1"])

    assert code == 1
    assert "没有可用于训练的有效音频" in capsys.readouterr().err


def test_fitprep_main_bad_usage_returns_usage_code(capsys):
    assert training.fitprep_main(["fitprep", "e", "48k", "v2"]) == 2
    assert training.fitprep_main(["fitprep", "e", "48k", "v2", "maybe"]) == 2
    assert training.fitprep_main(["precheck", "a", "b"]) == 2
    assert training.fitprep_main([]) == 2
    assert "用法" in capsys.readouterr().err


def test_precheck_subcommand_passes_and_fails(capsys):
    exp_dir = paths.LOGS_DIR / "mi-test"
    _fabricate_features(exp_dir)
    assert training.fitprep_main(["precheck", "mi-test"]) == 0
    assert "数据切分产物校验通过" in capsys.readouterr().out

    empty = paths.LOGS_DIR / "empty-exp"
    empty.mkdir(parents=True)
    assert training.fitprep_main(["precheck", "empty-exp"]) == 1
    assert "数据切分没有生成有效训练音频" in capsys.readouterr().err


def test_fitprep_module_resolvable_via_dash_m():
    """pipeline 的 cmd 是 `python -m server.api.training fitprep ...`（cwd=仓库根）：
    冒烟验证包路径可解析、模块可独立导入（usage 分支即可，不触真实目录）。"""
    out = subprocess.run(
        [sys.executable, "-m", "server.api.training"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 2, out.stderr
    assert "用法" in out.stderr


# ---------------------------------------------------------------------------
# 5. 任务查询 / 取消 / 队列
# ---------------------------------------------------------------------------


def test_submit_while_active_returns_queued_with_position(stub, client):
    stub.snapshots["existing"] = _snapshot("existing", name="pipeline")
    training._TASK_META["existing"] = {"exp_name": "other-exp"}

    resp = client.post("/api/train/fit", json=FIT_BODY)

    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "task-1"
    assert body["queued"] is True
    assert body["queue_position"] == 1


def test_submit_when_idle_returns_not_queued(stub, client):
    resp = client.post("/api/train/fit", json=FIT_BODY)

    assert resp.status_code == 200
    body = resp.json()
    assert body["queued"] is False
    assert body["queue_position"] is None


def test_same_exp_conflict_returns_409(stub, client):
    """同名实验（运行中）拒绝提交：防共写 logs/{exp}，是唯一保留的 409。"""
    stub.snapshots["existing"] = _snapshot("existing", name="pipeline")
    training._TASK_META["existing"] = {"exp_name": "mi-test"}

    resp = client.post("/api/train/fit", json=FIT_BODY)

    assert resp.status_code == 409
    assert "mi-test" in resp.json()["detail"]
    assert stub.created == []  # 冲突在登记之前拒绝


def test_same_exp_conflict_covers_pending_tasks(stub, client):
    stub.snapshots["existing"] = _snapshot(
        "existing", name="fit", state="pending", queue_position=1
    )
    training._TASK_META["existing"] = {"exp_name": "mi-test"}

    assert client.post("/api/train/fit", json=FIT_BODY).status_code == 409


def test_terminal_same_exp_does_not_conflict(stub, client):
    stub.snapshots["existing"] = _snapshot(
        "existing", name="fit", state="failed", error="x"
    )
    training._TASK_META["existing"] = {"exp_name": "mi-test"}

    assert client.post("/api/train/fit", json=FIT_BODY).status_code == 200


def test_same_exp_conflict_applies_to_all_training_endpoints(stub, client, tmp_path):
    """同名互斥覆盖 4 步与一键；数据集目录只对 preprocess/pipeline 校验。"""
    dataset = tmp_path / "ds"
    dataset.mkdir()
    stub.snapshots["existing"] = _snapshot("existing", name="fit", state="pending")
    training._TASK_META["existing"] = {"exp_name": "mi-test"}

    assert client.post("/api/train/preprocess", json={
        "exp_name": "mi-test", "dataset_dir": str(dataset)
    }).status_code == 409
    assert client.post("/api/train/extract", json={"exp_name": "mi-test"}).status_code == 409
    assert client.post("/api/train/index", json={"exp_name": "mi-test"}).status_code == 409
    assert client.post("/api/train/pipeline", json={
        **FIT_BODY, "dataset_dir": str(dataset)
    }).status_code == 409


# ---------------------------------------------------------------------------
# 1.5 实验名存在性检查（自选实验名防冲突；只挂 pipeline，续训走 allow_existing）
# ---------------------------------------------------------------------------


def test_exp_name_exists_endpoint_reports_absent_dir(client):
    resp = client.get("/api/train/exp-name/exists", params={"name": "mi-test"})

    assert resp.status_code == 200
    assert resp.json() == {"exists": False}


def test_exp_name_exists_endpoint_reports_present_dir(client):
    (paths.LOGS_DIR / "mi-test").mkdir(parents=True)

    resp = client.get("/api/train/exp-name/exists", params={"name": "mi-test"})

    assert resp.status_code == 200
    assert resp.json() == {"exists": True}


@pytest.mark.parametrize("exp_name", ["", "../evil", "a/b", "a b", "a$b", "a`b"])
def test_exp_name_exists_endpoint_rejects_invalid_name(client, exp_name):
    """非法名与提交接口同一张表（_check_exp_name）：前端输入时即时提示用。"""
    resp = client.get("/api/train/exp-name/exists", params={"name": exp_name})

    assert resp.status_code == 400
    assert "实验名非法" in resp.json()["detail"]


def test_pipeline_rejects_existing_exp_dir(stub, client, tmp_path):
    """自选实验名撞上已有产物目录 → 409，不创建任务：防误覆盖与无意识续训。"""
    (paths.LOGS_DIR / "mi-test").mkdir(parents=True)
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    resp = client.post(
        "/api/train/pipeline", json={**FIT_BODY, "dataset_dir": str(dataset)}
    )

    assert resp.status_code == 409
    assert "mi-test" in resp.json()["detail"]
    assert stub.created == []


def test_pipeline_allow_existing_permits_resume(stub, client, tmp_path):
    """allow_existing=True 是显式续训意图（任务历史「重新提交」路径），同名目录放行。"""
    (paths.LOGS_DIR / "mi-test").mkdir(parents=True)
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    resp = client.post(
        "/api/train/pipeline",
        json={**FIT_BODY, "dataset_dir": str(dataset), "allow_existing": True},
    )

    assert resp.status_code == 200
    assert stub.created[0]["name"] == "pipeline"


def test_exp_dir_existence_check_only_applies_to_pipeline(stub, client, tmp_path):
    """存在性检查不外溢到分步端点：fit/preprocess/extract 操作已存在实验是续训语义，
    Models 页「重建索引」也依赖已存在实验——它们的 409 只来自活动任务互斥。
    （stub 会把上一请求创建的任务记为 running，断言间须清掉活动任务状态。）"""
    (paths.LOGS_DIR / "mi-test").mkdir(parents=True)
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    stub.snapshots.clear()
    training._TASK_META.clear()
    assert client.post("/api/train/preprocess", json={
        "exp_name": "mi-test", "dataset_dir": str(dataset)
    }).status_code == 200

    stub.snapshots.clear()
    training._TASK_META.clear()
    assert client.post("/api/train/extract", json={"exp_name": "mi-test"}).status_code == 200

    stub.snapshots.clear()
    training._TASK_META.clear()
    assert client.post("/api/train/fit", json=FIT_BODY).status_code == 200

    stub.snapshots.clear()
    training._TASK_META.clear()
    assert client.post("/api/train/index", json={"exp_name": "mi-test"}).status_code == 200


def test_real_manager_second_submission_queues_and_same_exp_conflicts(
    client, monkeypatch, tmp_path
):
    """真 TaskManager 触发路径：运行中提交不同实验 → 200 排队；同名（运行中或
    排队中）→ 409。产物目录造齐让 setup 快速通过、任务保持 running。"""
    _fabricate_features(paths.LOGS_DIR / "mi-test")
    _fabricate_features(paths.LOGS_DIR / "second-exp")
    manager = TaskManager(
        poll_interval=0.01, popen=lambda cmd, **k: _FakeProcess(0, delay=10**6)
    )
    monkeypatch.setattr(training, "task_manager", manager)
    monkeypatch.setattr(training, "_TASK_META", {})
    try:
        first = client.post("/api/train/fit", json=FIT_BODY)
        assert first.status_code == 200
        wait_until(
            lambda: manager.get_task(first.json()["task_id"])["state"] == "running"
        )

        other = dict(FIT_BODY, exp_name="second-exp")
        resp = client.post("/api/train/fit", json=other)
        assert resp.status_code == 200
        assert resp.json()["queued"] is True
        queued_id = resp.json()["task_id"]
        wait_until(lambda: manager.get_task(queued_id)["state"] == "pending")

        assert client.post("/api/train/fit", json=other).status_code == 409  # 与排队同名
        assert client.post("/api/train/fit", json=FIT_BODY).status_code == 409  # 与运行中同名
    finally:
        manager.dispose()


def test_delete_tasks_clears_queue(stub, client):
    resp = client.delete("/api/tasks")

    assert resp.status_code == 200
    assert stub.cleared == 1
    assert resp.json() == {"stopped_task": None, "cancelled_pending": 0}


def test_create_records_definition_and_exp_meta(stub, client):
    resp = client.post("/api/train/fit", json=FIT_BODY)
    task_id = resp.json()["task_id"]

    created = stub.created[-1]
    assert created["definition"]["kind"] == "fit"
    assert created["definition"]["body"]["exp_name"] == "mi-test"
    assert created["definition"]["body"]["batch_size"] == 8
    assert training._TASK_META[task_id]["exp_name"] == "mi-test"


def test_pipeline_definition_carries_stages(stub, client, tmp_path):
    dataset = tmp_path / "ds"
    dataset.mkdir()

    resp = client.post(
        "/api/train/pipeline",
        json={**FIT_BODY, "dataset_dir": str(dataset), "batch_size": None},
    )

    created = stub.created[-1]
    assert created["definition"]["kind"] == "pipeline"
    assert created["definition"]["body"]["exp_name"] == "mi-test"
    assert created["definition"]["pipeline_stages"] == PIPELINE_STAGES


def test_get_task_returns_snapshot_with_computed_progress(stub, client):
    stub.snapshots["task-1"] = _snapshot(
        "task-1", logs=["INFO:mi-test:训练轮次：10 [50%]"]
    )
    training._TASK_META["task-1"] = {"total_epoch": 20}

    resp = client.get("/api/tasks/task-1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "running"
    # 进度按读时解析计算：((10 - 1) + 50/100) / 20
    assert body["progress"] == pytest.approx(0.475)
    assert body["logs"] == ["INFO:mi-test:训练轮次：10 [50%]"]


@pytest.mark.parametrize(
    "line, expected_progress, current",
    [
        ("[数据切分] 进度：12/34 | a.wav", 12 / 34, "a.wav"),
        ("[F0提取] 进度：3/9 | 成功：2 | 跳过：1 | b.flac", 3 / 9, "b.flac"),
        ("[HuBERT特征] 进度：7/10 | 成功：7 | 失败：0 | c.wav.npy | (1, 768)", 0.7, "c.wav.npy"),
        ("[Data slicing] Progress: 5/20 | d.wav", 0.25, "d.wav"),
        ("[HuBERT features] Progress: 1/4 | Success: 1 | Failed: 0 | e.wav | (1, 768)", 0.25, "e.wav"),
        ("[索引训练] 写入进度：5/9", 5 / 9, None),
    ],
)
def test_stage_progress_lines_drive_progress(stub, client, line, expected_progress, current):
    """design §3.5：切分/提取/索引阶段按脚本自身的进度行给进度 + 当前文件名。"""
    stub.snapshots["task-1"] = _snapshot("task-1", name="preprocess", logs=[line])

    body = client.get("/api/tasks/task-1").json()

    assert body["progress"] == pytest.approx(expected_progress)
    assert body["current"] == current


def test_stage_progress_zero_total_falls_back_to_raw(stub, client):
    stub.snapshots["task-1"] = _snapshot(
        "task-1", logs=["[数据切分] 进度：0/0 | a.wav"]
    )

    body = client.get("/api/tasks/task-1").json()

    assert body["progress"] is None and body["current"] is None


def test_fit_epoch_lines_win_over_stage_lines(stub, client):
    """pipeline 训练段开始后用 epoch 换算，不再回头用切分阶段的进度行。"""
    stub.snapshots["task-1"] = _snapshot(
        "task-1",
        name="pipeline",
        logs=["[数据切分] 进度：12/34 | a.wav", "INFO:mi-test:训练轮次：3 [50%]"],
    )
    training._TASK_META["task-1"] = {"total_epoch": 10}

    body = client.get("/api/tasks/task-1").json()

    assert body["progress"] == pytest.approx((2 + 0.5) / 10)
    assert body["current"] is None  # 训练阶段没有“当前文件”语义


def test_stage_progress_clamped_when_done_exceeds_total(stub, client):
    stub.snapshots["task-1"] = _snapshot(
        "task-1", logs=["[数据切分] 进度：40/34 | a.wav"]
    )

    assert client.get("/api/tasks/task-1").json()["progress"] == 1.0


def test_index_success_reports_full_progress(stub, client):
    stub.snapshots["task-1"] = _snapshot("task-1", name="index", state="success")

    body = client.get("/api/tasks/task-1").json()

    assert body["progress"] == 1.0 and body["current"] is None


def test_failed_preprocess_keeps_partial_progress(stub, client):
    stub.snapshots["task-1"] = _snapshot(
        "task-1",
        name="preprocess",
        state="failed",
        error="boom",
        logs=["[数据切分] 进度：12/34 | a.wav"],
    )

    body = client.get("/api/tasks/task-1").json()

    assert body["progress"] == pytest.approx(12 / 34)
    assert body["current"] == "a.wav"


def test_get_task_without_meta_keeps_raw_progress(stub, client):
    stub.snapshots["task-1"] = _snapshot("task-1", name="preprocess", progress=None)

    resp = client.get("/api/tasks/task-1")

    assert resp.status_code == 200
    assert resp.json()["progress"] is None  # 非训练阶段无解析依据，透传原值


def test_get_task_unknown_returns_404(stub, client):
    resp = client.get("/api/tasks/no-such-id")

    assert resp.status_code == 404


def test_list_tasks_projects_compact_fields(stub, client):
    stub.snapshots = {
        "a": _snapshot("a", name="preprocess", progress=None),
        "b": _snapshot("b", name="fit", state="failed", error="boom", logs=["x"]),
        "c": _snapshot("c", name="pipeline", state="pending", queue_position=2),
    }
    training._TASK_META["b"] = {"total_epoch": 10, "exp_name": "second"}
    stub.snapshots["b"]["logs"] = ["INFO:mi-test:训练轮次：5 [40%]"]

    resp = client.get("/api/tasks")

    assert resp.status_code == 200
    body = resp.json()
    assert [(row["id"], row["state"], row["queue_position"], row["exp_name"], row["history"]) for row in body] == [
        ("a", "running", None, None, False),
        ("b", "failed", None, "second", False),
        ("c", "pending", 2, None, False),
    ]
    # 进度仍按读时解析计算
    assert body[1]["progress"] == pytest.approx(((5 - 1) + 0.4) / 10)
    # 内存行的扩展投影：params/kind/pipeline_stages 来自 definition 与元数据
    assert body[1]["kind"] is None and body[1]["params"] == {}
    assert body[1]["created_at"] == 1.0 and body[1]["finished_at"] is None


def test_delete_task_returns_202_and_cancels(stub, client):
    stub.snapshots["task-1"] = _snapshot("task-1")

    resp = client.delete("/api/tasks/task-1")

    assert resp.status_code == 202
    assert stub.cancelled == ["task-1"]


def test_delete_unknown_task_returns_404(stub, client):
    assert client.delete("/api/tasks/no-such-id").status_code == 404


def test_lifespan_disposes_task_manager_on_shutdown(monkeypatch):
    """硬性要求⑤：shutdown 必须调 task_manager.dispose()，否则 daemon 工作线程被硬杀、
    训练子进程孤儿化。"""
    from server import main as main_module

    calls = []
    monkeypatch.setattr(
        main_module.task_manager, "dispose", lambda *a, **k: calls.append(True)
    )

    with TestClient(create_app()):
        assert calls == []  # 启动阶段不清理
    assert calls == [True]


# ---------------------------------------------------------------------------
# pipeline 全流程加权进度（单一刻度单调递增，阶段切换不再归零）
# ---------------------------------------------------------------------------

PIPELINE_STAGES = ["preprocess", "preprocess", "extract", "extract", "fit", "fit", "index"]


def test_pipeline_progress_weighted_across_stages(stub, client):
    """切分段：base 0 + 阶段内进度 × 5%。"""
    stub.snapshots["task-1"] = _snapshot(
        "task-1", name="pipeline", logs=["[数据切分] 进度：1/2 | 9.wav"], current_cmd=1
    )
    training._TASK_META["task-1"] = {
        "total_epoch": 20,
        "pipeline_stages": list(PIPELINE_STAGES),
    }

    body = client.get("/api/tasks/task-1").json()

    assert body["progress"] == pytest.approx(0.05 * 0.5)
    assert body["current"] == "9.wav"


def test_pipeline_progress_fit_segment_starts_at_20_percent(stub, client):
    """训练段开始：base = 切分 5% + 提取 15% = 20%，epoch 进度在剩余 75% 里爬升，
    不再从 0 归零（epoch1 0% → 0.2；epoch 11 50% → 0.2 + 0.75 × (10.5/20)）。"""
    training._TASK_META["task-1"] = {
        "total_epoch": 20,
        "pipeline_stages": list(PIPELINE_STAGES),
    }

    stub.snapshots["task-1"] = _snapshot(
        "task-1",
        name="pipeline",
        logs=["[数据切分] 进度：1/1 | 9.wav", "INFO:mi-test:Training epoch: 1 [0%]"],
        current_cmd=6,
    )
    body = client.get("/api/tasks/task-1").json()
    assert body["progress"] == pytest.approx(0.20)
    assert body["current"] is None

    stub.snapshots["task-1"] = _snapshot(
        "task-1",
        name="pipeline",
        logs=["INFO:mi-test:训练轮次：11 [50%]"],
        current_cmd=6,
    )
    body = client.get("/api/tasks/task-1").json()
    assert body["progress"] == pytest.approx(0.20 + 0.75 * (10.5 / 20))


def test_pipeline_progress_is_monotonic_across_f0_batches(stub, client):
    """F0 多批次每批 done/total 从低处重新计数：单调钳位不让 bar 倒退。"""
    training._TASK_META["task-1"] = {
        "total_epoch": 20,
        "pipeline_stages": list(PIPELINE_STAGES),
    }
    stub.snapshots["task-1"] = _snapshot(
        "task-1", name="pipeline", logs=["[F0提取] 进度：1/1 | a.wav"], current_cmd=3
    )
    first = client.get("/api/tasks/task-1").json()["progress"]
    assert first == pytest.approx(0.05 + 0.15)  # 单文件批次直接满格

    # 下一批重新从 1/2 计数：不钳位会倒退到 12.5%
    stub.snapshots["task-1"] = _snapshot(
        "task-1", name="pipeline", logs=["[F0提取] 进度：1/2 | b.wav"], current_cmd=3
    )
    second = client.get("/api/tasks/task-1").json()["progress"]
    assert second == pytest.approx(first)


def test_pipeline_progress_before_first_cmd_is_zero(stub, client):
    training._TASK_META["task-1"] = {
        "total_epoch": 20,
        "pipeline_stages": list(PIPELINE_STAGES),
    }
    stub.snapshots["task-1"] = _snapshot("task-1", name="pipeline", logs=[], current_cmd=None)

    body = client.get("/api/tasks/task-1").json()

    assert body["progress"] == 0.0


def test_pipeline_stage_table_registered_on_create(stub, client, monkeypatch, tmp_path):
    """建任务即登记 stages 表（与 cmds 等长、if_f0=False 时少一段 extract）。"""
    monkeypatch.setattr(training, "resolve_is_half", lambda: False)
    monkeypatch.setattr(commands, "_resolve_device", lambda: "cpu")
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    resp = client.post(
        "/api/train/pipeline",
        json={**FIT_BODY, "if_f0": False, "dataset_dir": str(dataset)},
    )

    assert resp.status_code == 200
    meta = training._TASK_META[resp.json()["task_id"]]
    assert meta["pipeline_stages"] == [
        "preprocess",
        "preprocess",
        "extract",
        "fit",
        "fit",
        "index",
    ]
    assert len(meta["pipeline_stages"]) == len(stub.created[0]["cmds"])
