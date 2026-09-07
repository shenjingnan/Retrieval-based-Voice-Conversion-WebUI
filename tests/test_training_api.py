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
from server.tasks import TaskConflictError, TaskManager

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
CPU = os.cpu_count()


# ---------------------------------------------------------------------------
# 替身与夹具
# ---------------------------------------------------------------------------


class StubTaskManager:
    """记录 create_task 入参的替身；快照由测试直接塞进 snapshots。"""

    def __init__(self):
        self.created = []  # create_task 入参（按创建序）
        self.order = []  # task_id 创建序
        self.snapshots = {}
        self.cancelled = []
        self.conflict = None  # 置为异常实例 → create_task 抛出（模拟互斥）

    def create_task(self, name, cmds, log_path, truncate: bool = True, *, setup=None):
        if self.conflict is not None:
            raise self.conflict
        self._seq = getattr(self, "_seq", 0) + 1
        task_id = "task-%d" % self._seq
        self.created.append(
            {
                "id": task_id,
                "name": name,
                "cmds": list(cmds),
                "log_path": log_path,
                "truncate": truncate,
                "setup": setup,
            }
        )
        self.order.append(task_id)
        self.snapshots[task_id] = _snapshot(task_id, name)
        return task_id

    def get_task(self, task_id):
        return self.snapshots.get(task_id)

    def list_tasks(self):
        return [self.snapshots[task_id] for task_id in self.order]

    def cancel(self, task_id):
        self.cancelled.append(task_id)
        return True


def _snapshot(task_id, name="fit", state="running", progress=None, error=None, logs=()):
    return {
        "id": task_id,
        "name": name,
        "state": state,
        "progress": progress,
        "error": error,
        "cmds": ["cmd"],
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
    assert resp.json() == {"task_id": "task-1"}
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
    assert call["setup"] is None  # 任务级 setup 会在首个 cmd 前误判产物校验，pipeline 禁用
    # 前端进度条需要 total_epoch：任务元数据随创建登记
    assert training._TASK_META["task-1"] == {"total_epoch": 20}


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
    ["", "../evil", "a/b", "..", ".", 'a"b', "a\\b", "a b", "a\tb", "a\nb", "a$b", "a`b"],
)
def test_invalid_exp_name_rejected(stub, client, tmp_path, exp_name):
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
# 5. 任务查询 / 取消 / 互斥
# ---------------------------------------------------------------------------


def test_conflict_maps_to_409(stub, client):
    stub.conflict = TaskConflictError("已有训练任务在运行")

    resp = client.post("/api/train/fit", json=FIT_BODY)

    assert resp.status_code == 409
    assert "已有训练任务在运行" in resp.json()["detail"]


def test_conflict_with_real_manager_maps_to_409(client, monkeypatch, tmp_path):
    """真 TaskManager 的互斥也走 409（Stub 只验证映射，这里验证触发路径）。"""
    manager = TaskManager(
        poll_interval=0.01, popen=lambda cmd, **k: _FakeProcess(0, delay=10**6)
    )
    monkeypatch.setattr(training, "task_manager", manager)
    monkeypatch.setattr(training, "_TASK_META", {})
    try:
        first = manager.create_task("fit", ["cmd"], tmp_path / "a.log")
        wait_until(lambda: manager.get_task(first)["state"] == "running")

        resp = client.post("/api/train/fit", json=FIT_BODY)

        assert resp.status_code == 409
    finally:
        manager.dispose()


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
    }
    stub.order = ["a", "b"]
    training._TASK_META["b"] = {"total_epoch": 10}
    stub.snapshots["b"]["logs"] = ["INFO:mi-test:训练轮次：5 [40%]"]

    resp = client.get("/api/tasks")

    assert resp.status_code == 200
    body = resp.json()
    assert body == [
        {"id": "a", "name": "preprocess", "state": "running", "progress": None, "error": None},
        {
            "id": "b",
            "name": "fit",
            "state": "failed",
            "progress": pytest.approx(((5 - 1) + 0.4) / 10),
            "error": "boom",
        },
    ]


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
