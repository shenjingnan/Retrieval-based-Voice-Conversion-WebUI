"""atomic_torch_save：成功时原子替换目标，失败时不留临时文件、不损坏旧检查点。"""
import glob
import os

import pytest
import torch

from tools.file_io import atomic_torch_save


def _tmp_files(directory):
    return glob.glob(os.path.join(directory, "*.tmp"))


def test_save_creates_missing_target(tmp_path):
    target = str(tmp_path / "G_60.pth")

    atomic_torch_save({"model": torch.zeros(2, 2), "iteration": 5}, target)

    loaded = torch.load(target, weights_only=True)
    assert loaded["iteration"] == 5
    assert loaded["model"].shape == (2, 2)
    assert _tmp_files(str(tmp_path)) == []


def test_save_replaces_existing_target(tmp_path):
    target = str(tmp_path / "G_120.pth")
    torch.save({"model": torch.ones(1), "iteration": 5}, target)

    atomic_torch_save({"model": torch.zeros(1), "iteration": 10}, target)

    loaded = torch.load(target, weights_only=True)
    assert loaded["iteration"] == 10
    assert loaded["model"].tolist() == [0.0]
    assert _tmp_files(str(tmp_path)) == []


def test_failed_save_keeps_target_and_cleans_tmp(tmp_path):
    target = str(tmp_path / "G_180.pth")
    old = {"model": torch.ones(1), "iteration": 5}
    torch.save(old, target)

    # lambda 无法 pickle：torch.save 在写临时文件途中抛错
    with pytest.raises(Exception):
        atomic_torch_save({"model": lambda: None}, target)

    assert torch.load(target, weights_only=True)["iteration"] == 5  # 旧检查点完好
    assert _tmp_files(str(tmp_path)) == []  # 不留半截临时文件


@pytest.mark.skipif(os.name != "nt", reason="Windows 下替换被占用的文件会失败")
def test_replace_target_held_open_by_reader_keeps_old_file(tmp_path):
    """复现本次事故场景：目标被残留进程占用时，宁可报错也绝不写坏旧文件。"""
    target = str(tmp_path / "G_240.pth")
    atomic_torch_save({"model": torch.ones(1), "iteration": 5}, target)

    with open(target, "rb"):  # CPython 打开文件不带 FILE_SHARE_DELETE
        with pytest.raises(PermissionError):
            atomic_torch_save({"model": torch.zeros(1), "iteration": 10}, target)

    assert torch.load(target, weights_only=True)["iteration"] == 5
    assert _tmp_files(str(tmp_path)) == []
