# -*- coding: utf-8 -*-
"""
backends.py — ChatGarment 推理后端适配器（三种可插拔模式）。

设计目的：ComfyUI 节点不关心模型在哪跑，只认统一接口：
    generate(image_path=None, prompt="") -> specification.json (dict)

四种后端：
  mock      本机零依赖，返回内置样例，用于验证 UI/转换链路是否接通。
  http      连一个跑在云 GPU / 局域网的推理服务（见 garment_api_server.py），
            POST 图片+文本，返回 specification JSON。
  local     在本机 conda(chatgarment) 环境里跑官方推理脚本（需 24G+ 显存）。
  aipparel  在 RunningHub / 云 GPU 上跑 AIpparel 官方推理链，权重从模型库
            (ComfyUI/models/checkpoints) 选择，输出 specification JSON。

项目要求：ChatGarment 真正推理走 scripts/evaluate_garment_v2_imggen_1float.py
（LLaVA-1.5-7B + deepspeed + flash-attn2 + bf16），直接吐的是「设计参数 JSON」，
必须再经 GarmentCodeRC 的 MetaGarment.assembly() 展开成 specification.json
才有真正的 2D 板片坐标。http/local 后端负责这两步，节点拿到的始终是
最终 specification.json。
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))


def _comfy_output_dir():
    """在 ComfyUI 内拿到 output 目录；脱离 ComfyUI 单测时退回工作区。"""
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return os.path.join(HERE, "output")


def _load_sample():
    with open(os.path.join(HERE, "sample_specification.json"), "r", encoding="utf-8") as f:
        return json.load(f)


class MockBackend:
    """零依赖后端：返回内置样例，用来验证 ComfyUI→转换→Atelier 全链路。"""
    name = "mock"

    def generate(self, image_path=None, prompt="", **kw):
        spec = _load_sample()
        spec.setdefault("_mock", {"prompt": prompt, "image": os.path.basename(image_path) if image_path else None})
        return spec


class HttpBackend:
    """连云端/局域网推理服务。服务实现见 work/garment_plugin/garment_api_server.py。

    协议: POST /generate
      multipart/form-data: image=<file>  prompt=<str>  mode=<img|txt>
      -> application/json: {"specification": {...}, "design_params": {...}, "error": null}
    """
    name = "http"

    def __init__(self, endpoint="http://127.0.0.1:8199", timeout=900):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def generate(self, image_path=None, prompt="", mode="img", **kw):
        boundary = "----GarmentBoundary7MA4YWxk"
        parts = []

        def field(name, value):
            parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" %
                          (boundary, name, value)).encode("utf-8"))

        field("prompt", prompt or "")
        field("mode", mode)
        if image_path and os.path.isfile(image_path):
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            fname = os.path.basename(image_path)
            parts.append(("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"%s\"\r\n"
                          "Content-Type: application/octet-stream\r\n\r\n" % (boundary, fname)).encode("utf-8"))
            parts.append(img_bytes)
            parts.append(b"\r\n")
        parts.append(("--%s--\r\n" % boundary).encode("utf-8"))
        body = b"".join(parts)

        req = urllib.request.Request(
            self.endpoint + "/generate", data=body,
            headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("error"):
            raise RuntimeError("推理服务返回错误: %s" % data["error"])
        spec = data.get("specification")
        if not spec:
            raise RuntimeError("推理服务未返回 specification")
        return spec


class LocalBackend:
    """本机 conda(chatgarment) 环境跑官方推理 + GarmentCodeRC 展开。

    需要用户已装好 ChatGarment 环境（见 README），并配置：
      conda_python : conda 环境的 python.exe
      chatgarment_dir / garmentcode_dir : 两个仓库根目录
    """
    name = "local"

    def __init__(self, conda_python="", chatgarment_dir="", garmentcode_dir="", timeout=3600):
        self.conda_python = conda_python
        self.chatgarment_dir = chatgarment_dir
        self.garmentcode_dir = garmentcode_dir
        self.timeout = timeout

    def generate(self, image_path=None, prompt="", **kw):
        if not all([self.conda_python, self.chatgarment_dir, self.garmentcode_dir]):
            raise RuntimeError("LocalBackend 需要配置 conda_python / chatgarment_dir / garmentcode_dir")
        # 调用一个包装脚本：先跑 ChatGarment 出设计参数，再跑 GarmentCodeRC 展开
        runner = os.path.join(HERE, "local_runner.py")
        cmd = [self.conda_python, runner,
               "--chatgarment", self.chatgarment_dir,
               "--garmentcode", self.garmentcode_dir,
               "--prompt", prompt or ""]
        if image_path:
            cmd += ["--image", image_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError("本地推理失败: %s" % proc.stderr[-1500:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        if out.get("error"):
            raise RuntimeError(out["error"])
        return out["specification"]


class AipparelBackend:
    """AIpparel 官方推理链（打包在 aipparel/ 内），面向 RunningHub 云端 GPU。

    权重解析顺序：
      1. weights_path 文本框里的绝对路径（若文件存在）；
      2. RunningHub/ComfyUI 模型库里的 checkpoint 名（models/checkpoints/）；
      3. 传给 weights 的绝对路径。
    """
    name = "aipparel"

    def __init__(self, weights="", weights_path="", version="liuhaotian/llava-v1.5-7b",
                 vision_tower="openai/clip-vit-large-patch14", precision="bf16",
                 python_executable="", max_new_tokens=2100, timeout=7200,
                 include_prompt=False, cache_dir=""):
        self.weights = weights
        self.weights_path = weights_path
        self.version = version
        self.vision_tower = vision_tower
        self.precision = precision
        if python_executable:
            self.python_executable = python_executable
        else:
            # 如果已经用 setup 节点建好隔离环境，自动使用它，避免误用 ComfyUI 的全局 python
            venv_py = os.path.join(HERE, ".venv_aipparel", "Scripts" if os.name == "nt" else "bin",
                                   "python.exe" if os.name == "nt" else "python")
            self.python_executable = venv_py if os.path.isfile(venv_py) else sys.executable
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.include_prompt = include_prompt
        self.cache_dir = cache_dir

    def _resolve_weights(self):
        for candidate in (self.weights_path, self.weights):
            if candidate and os.path.isfile(candidate):
                return os.path.abspath(candidate)
        name = self.weights_path or self.weights
        if not name:
            raise RuntimeError("请在模型权重下拉里选择上传的 aipparel_pretrained 文件")
        try:
            import folder_paths
            full = folder_paths.get_full_path("checkpoints", name)
            if full and os.path.isfile(full):
                return full
            available = folder_paths.get_filename_list("checkpoints")
            raise RuntimeError(
                "模型库里找不到 %s。当前可见的 checkpoint: %s"
                % (name, ", ".join(available[:20]) or "(空)")
            )
        except ImportError as e:
            raise RuntimeError("未找到权重文件 %s，且当前环境没有 ComfyUI 模型库" % name) from e

    def generate(self, image_path=None, prompt="", mode="img", **kw):
        weights = self._resolve_weights()
        stamp = time.strftime("%Y%m%d_%H%M%S") + "_%04d" % (os.getpid() % 10000)
        out_dir = os.path.join(_comfy_output_dir(), "garment_pattern", "aipparel_runs", stamp)
        manifest = os.path.join(out_dir, "result.json")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            self.python_executable, os.path.join(HERE, "aipparel_runner.py"),
            "--weights", weights,
            "--out-dir", out_dir,
            "--manifest", manifest,
            "--version", self.version,
            "--vision-tower", self.vision_tower,
            "--precision", self.precision,
            "--max-new-tokens", str(self.max_new_tokens),
        ]
        if self.cache_dir:
            cmd += ["--cache-dir", self.cache_dir]
        if image_path and os.path.isfile(image_path) and mode != "txt":
            cmd += ["--image", os.path.abspath(image_path)]
        if prompt:
            cmd += ["--prompt", prompt]
        if self.include_prompt:
            cmd += ["--include-prompt"]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, env=env)
        if not os.path.isfile(manifest):
            raise RuntimeError(
                "AIpparel 推理未产出结果 (returncode=%d)。\nstdout:\n%s\nstderr:\n%s"
                % (proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:])
            )
        with open(manifest, "r", encoding="utf-8") as f:
            result = json.load(f)
        if result.get("error"):
            raise RuntimeError("AIpparel 推理失败: %s\nstderr:\n%s"
                               % (result["error"], proc.stderr[-3000:]))
        spec = result.get("specification")
        if not spec:
            raise RuntimeError("AIpparel 推理结果里没有 specification")
        # 把运行目录信息带回去，方便排查和复现
        spec.setdefault("_aipparel", {
            "weights": os.path.basename(weights),
            "run_dir": out_dir,
            "spec_files": result.get("spec_files", []),
            "output_text": result.get("output_text", ""),
        })
        return spec


def get_backend(mode, **kw):
    mode = (mode or "mock").lower()
    if mode == "mock":
        return MockBackend()
    if mode == "http":
        return HttpBackend(**kw)
    if mode == "local":
        return LocalBackend(**kw)
    if mode == "aipparel":
        return AipparelBackend(**kw)
    raise ValueError("未知后端模式: %s" % mode)
