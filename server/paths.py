"""运行时路径常量。server 只在此处感知目录布局。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# config.json 模板目录：webui 的 configs.json_config 读的是同一批文件。
# 单独成常量是因为测试会把 ROOT 指到 tmp（隔离 assets/logs），而模板必须仍读真实仓库
CONFIGS_DIR = ROOT / "configs"
ASSETS = ROOT / "assets"
WEIGHTS_DIR = ASSETS / "weights"
# 索引统一放 assets/indices：训练侧 train_index.py 的 outside_index_root（webui.py:25 默认值，
# server.commands.INDEX_ROOT 由此派生）与 P1 /api/models 的扫描目录一致，训完即可配对。
# 注意：P1 的 /api/models 会跳过 *_spkidN 索引（无说话人上下文）；
# P3 引入多说话人时需带 speaker_id 重新选索引。
INDICES_DIR = ASSETS / "indices"
LOGS_DIR = ROOT / "logs"
# 用户上传的训练数据集根目录（server/api/datasets.py 维护其一级子目录）；
# 训练侧拿到的是其中某个子目录的绝对路径（作为 dataset_dir 传给 preprocess）
DATASETS_DIR = ROOT / "datasets"
STATIC_DIR = Path(__file__).resolve().parent / "static"  # 前端构建产物，可能不存在


def queue_file() -> Path:
    """训练任务队列持久化文件（server/task_store.py 的默认落点，logs/ 已 gitignore）。
    用函数而非常量：测试把 ROOT monkeypatch 到 tmp 后此处跟随，测试不触碰真实 logs/。"""
    return ROOT / "logs" / "task_queue.json"


def history_file() -> Path:
    """训练历史记录文件（TaskHistoryStore 默认落点，同 queue_file 的函数式理由）。"""
    return ROOT / "logs" / "task_history.json"
