#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
garment_api_server.py — 部署在云 GPU 上的推理服务，供 ComfyUI 插件的 http 后端调用。

零第三方依赖（Python 标准库 http.server），协议：
  GET  /health
  POST /generate   multipart/form-data: image=<file> prompt=<str> mode=<img|txt>
       -> {"specification": {...}, "design_params": {...}, "error": null}

职责（对应 ChatGarment 官方真实推理链路，均已实测核实）：
  mode=img: 图片 -> ChatGarment 两阶段推理 -> 设计参数 JSON
            -> GarmentCodeRC 展开 -> specification.json（含真正 2D 板片坐标）
  mode=txt: 文本 -> 设计参数 -> specification.json

【重要边界】这是「服务骨架 + 接入契约」，不是开箱即用的完整推理。
真实部署需你在云 GPU 上完成（TODO 已标出两个调用点）：
  1. 下载 ChatGarment 权重（SharePoint）
  2. 装 GarmentCodeRC 并改 sys.path / 软链 assets
  3. 按实际路径改下面的命令与 import
generate_design_params / design_to_specification 当前返回演示数据，
保证 ComfyUI 插件的 http 链路可先联调；接模型后节点无需改动。
"""
import json
import os
import re
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHATGARMENT_DIR = os.environ.get("CHATGARMENT_DIR", "/workspace/ChatGarment")
GARMENTCODE_DIR = os.environ.get("GARMENTCODE_DIR", "/workspace/GarmentCodeRC")
HERE = os.path.dirname(os.path.abspath(__file__))


def generate_design_params(image_path, prompt, mode):
    """TODO(云端): 调 ChatGarment 官方推理，返回设计参数 JSON。

    参考命令（README + scripts/evaluate_garment_v2_imggen_1float.py）:
      cd $CHATGARMENT_DIR
      deepspeed scripts/evaluate_garment_v2_imggen_1float.py \
        --model_name_or_path liuhaotian/llava-v1.5-7b \
        --data_path_eval <含图片的目录> --image_folder ./ ...
    输出 runs/<exp>/<ds>_img_recon/vis_new/valid_garment_*/output.txt，
    最后一行 str(json_output) 即设计参数（形如 {"wholebody_garment": {...}}）。
    """
    return {
        "wholebody_garment": {
            "meta": {"style": "t-shirt"},
            "top": {"type": "t-shirt", "sleeve": {"type": "short"}},
        }
    }, {"note": "DEMO design params — 未接真实模型"}


def design_to_specification(design_params):
    """TODO(云端): 用 GarmentCodeRC 展开设计参数为 specification.json。

    参考 llava/garment_utils_v2.py::try_generate_garments 的真实调用：
      design = recursive_change_params(deepcopy(designs_config), design_pred)
      body = BodyParameters('./assets/bodies/mean_all.yaml')
      pattern = MetaGarment('x', body, design).assembly()
      folder = pattern.serialize(out, with_3d=False)
      -> <folder>/valid_garment_x_specification.json
    """
    sample = os.path.join(HERE, "samples", "shirt_mean_spec_sample.json")
    with open(sample, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_multipart(body, content_type):
    """极简 multipart 解析：只取 prompt/mode 文本字段与 image 文件。"""
    m = re.search(r"boundary=(.+)", content_type)
    if not m:
        return {}, {}
    boundary = m.group(1).strip().strip('"')
    fields, files = {}, {}
    for part in body.split(("--" + boundary).encode()):
        if not part or part in (b"\r\n", b"--\r\n"):
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        data = data.rstrip(b"\r\n")
        h = head.decode("utf-8", "ignore")
        name_m = re.search(r'name="([^"]+)"', h)
        if not name_m:
            continue
        name = name_m.group(1)
        if "filename=" in h:
            files[name] = data
        else:
            fields[name] = data.decode("utf-8", "ignore")
    return fields, files


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._json({"status": "ok", "chatgarment_dir": CHATGARMENT_DIR,
                        "garmentcode_dir": GARMENTCODE_DIR})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/generate"):
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            fields, files = _parse_multipart(body, self.headers.get("Content-Type", ""))
            mode = fields.get("mode", "img")
            prompt = fields.get("prompt", "")
            img_path = None
            if mode == "img" and "image" in files:
                fd, img_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                with open(img_path, "wb") as f:
                    f.write(files["image"])
            design, dbg = generate_design_params(img_path, prompt, mode)
            spec = design_to_specification(design)
            if img_path and os.path.exists(img_path):
                os.remove(img_path)
            self._json({"specification": spec, "design_params": dbg, "error": None})
        except Exception as e:
            self._json({"specification": None, "design_params": None, "error": str(e)}, 500)

    def log_message(self, fmt, *args):
        print("[server]", fmt % args)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8199"))
    print("Garment API server on 0.0.0.0:%d" % port)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
