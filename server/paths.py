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
STATIC_DIR = Path(__file__).resolve().parent / "static"  # 前端构建产物，可能不存在
