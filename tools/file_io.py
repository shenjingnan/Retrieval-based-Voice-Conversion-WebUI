import os


def read_text(path, errors="strict", newline=None):
    last_error = None
    for encoding in (None, "utf8", "gbk"):
        try:
            kwargs = {"errors": "strict", "newline": newline}
            if encoding is not None:
                kwargs["encoding"] = encoding
            with open(path, "r", **kwargs) as file:
                return file.read()
        except UnicodeDecodeError as error:
            last_error = error
    if errors != "strict":
        with open(path, "r", encoding="gbk", errors=errors, newline=newline) as file:
            return file.read()
    raise last_error


def atomic_torch_save(obj, path):
    """torch.save 的原子版：先写同目录临时文件，成功后 os.replace 原子替换目标。

    直接写目标时，写入中途被打断（进程被杀/磁盘干扰/锁冲突）会在目标处留下
    写了一半的损坏检查点，下次训练恢复时加载失败。临时文件方案保证目标要么
    是完整的旧文件、要么是完整的新文件；失败时清理临时文件并原样抛出。
    """
    import torch

    tmp_path = "%s.%d.tmp" % (path, os.getpid())
    try:
        with open(tmp_path, "wb") as file:
            torch.save(obj, file)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
