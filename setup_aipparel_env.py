#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_aipparel_env.py — 给 AIpparel 建隔离 Python 环境，避免污染 ComfyUI/RunningHub 全局依赖。

AIpparel 官方代码锁定 transformers==4.31.0，而 ComfyUI 环境通常带更新版本
（本机实测 5.2.0，且缺 svgpathtools）。这里用 --system-site-packages 继承宿主
torch/CUDA，只在 venv 内覆盖 transformers 等少数包，不动全局环境。
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VENV = os.path.join(HERE, ".venv_aipparel")


def venv_python(venv_dir=DEFAULT_VENV):
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def setup(venv_dir=DEFAULT_VENV, requirements=None, force=False):
    requirements = requirements or os.path.join(HERE, "aipparel_requirements.txt")
    py = venv_python(venv_dir)
    if force or not os.path.isfile(py):
        cmd = [sys.executable, "-m", "venv", "--system-site-packages", venv_dir]
        subprocess.run(cmd, check=True)
    subprocess.run([py, "-m", "pip", "install", "-r", requirements], check=True)
    out = subprocess.run(
        [py, "-c", "import transformers, torch; print(transformers.__version__, torch.__version__)"],
        capture_output=True, text=True, check=True,
    )
    return "venv=%s\ntransformers, torch = %s" % (py, out.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=DEFAULT_VENV)
    ap.add_argument("--requirements", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(setup(a.venv, a.requirements or None, a.force))


if __name__ == "__main__":
    main()
