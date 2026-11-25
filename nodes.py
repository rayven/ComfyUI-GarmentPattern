# -*- coding: utf-8 -*-
"""
nodes.py — ComfyUI 节点：AI 生成服装 → specification.json → 板片/缝纫 JSON。

节点链（从左到右）：
  1. GarmentGenerateNode      图/文 → specification.json（走 backends 三选一）
  2. GarmentSpecToPatternNode specification.json → pieces/sewings json（转换层）
  3. GarmentSpecPreviewNode   预览统计信息
  4. GarmentSaveNode          把结果 json 存到 ComfyUI/output 供下载

用户拿 pieces/sewings json 下载后，用 tools/atelier_rebuild_lite.py 等
在 Style3D Atelier 里画出板片与缝合线。
"""
import json
import os
import tempfile
import time

import torch
import numpy as np
from PIL import Image

import folder_paths

import spec_convert as C
from backends import get_backend

OUTPUT_DIR = os.path.join(folder_paths.get_output_directory(), "garment_pattern")

try:
    _CHECKPOINTS = folder_paths.get_filename_list("checkpoints")
except Exception:
    _CHECKPOINTS = []


def _tensor_to_temp_png(image_tensor):
    """ComfyUI IMAGE (B,H,W,C float 0..1) -> 临时 png 路径。"""
    if image_tensor is None:
        return None
    arr = image_tensor[0].detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="garment_input_")
    os.close(fd)
    img.save(path)
    return path


def _save_json(data, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, "%s_%s.json" % (name, ts))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


class GarmentGenerateNode:
    """图/文 → specification.json（ChatGarment / AIpparel）。"""

    @classmethod
    def INPUT_TYPES(cls):
        weights_input = ((_CHECKPOINTS or ["aipparel_pretrained.safetensors"]),
                         {"default": (_CHECKPOINTS or ["aipparel_pretrained.safetensors"])[0]})
        return {
            "required": {
                "mode": (["mock", "http", "local", "aipparel"], {"default": "mock"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "a short sleeve shirt"}),
                "http_endpoint": ("STRING", {"default": "http://127.0.0.1:8199"}),
                "weights": weights_input,
                "weights_path": ("STRING", {"default": "",
                                            "tooltip": "可选：直接填写云端权重绝对路径，优先于下拉"}),
                "llava_version": ("STRING", {"default": "liuhaotian/llava-v1.5-7b"}),
                "vision_tower": ("STRING", {"default": "openai/clip-vit-large-patch14"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "max_new_tokens": ("INT", {"default": 2100, "min": 64, "max": 4096}),
                "timeout": ("INT", {"default": 7200, "min": 60, "max": 86400}),
                "python_executable": ("STRING", {"default": "",
                                                 "tooltip": "可选：隔离环境 python.exe；留空用 ComfyUI 自带 python"}),
                "hf_cache_dir": ("STRING", {"default": ""}),
                "include_prompt": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("specification_json",)
    FUNCTION = "generate"
    CATEGORY = "GarmentPattern"

    def generate(self, mode, image=None, prompt="", http_endpoint="http://127.0.0.1:8199",
                 weights="aipparel_pretrained.safetensors", weights_path="",
                 llava_version="liuhaotian/llava-v1.5-7b",
                 vision_tower="openai/clip-vit-large-patch14", precision="bf16",
                 max_new_tokens=2100, timeout=7200, python_executable="",
                 hf_cache_dir="", include_prompt=False):
        img_path = _tensor_to_temp_png(image)
        kw = {}
        if mode == "http":
            kw["endpoint"] = http_endpoint
        elif mode == "aipparel":
            kw = dict(
                weights=weights, weights_path=weights_path, version=llava_version,
                vision_tower=vision_tower, precision=precision,
                max_new_tokens=max_new_tokens, timeout=timeout,
                python_executable=python_executable, cache_dir=hf_cache_dir,
                include_prompt=include_prompt,
            )
        backend = get_backend(mode, **kw)
        m = "img" if img_path else "txt"
        spec = backend.generate(image_path=img_path, prompt=prompt, mode=m)
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        return (json.dumps(spec, ensure_ascii=False),)


class GarmentSpecToPatternNode:
    """specification.json → Atelier pieces/sewings json。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "specification_json": ("STRING", {"multiline": True}),
            },
            "optional": {
                "curve_steps": ("INT", {"default": 12, "min": 2, "max": 40}),
                "sew_steps": ("INT", {"default": 12, "min": 2, "max": 40}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("atelier_garment_json",)
    FUNCTION = "convert"
    CATEGORY = "GarmentPattern"

    def convert(self, specification_json, curve_steps=12, sew_steps=12):
        spec = json.loads(specification_json)
        garment = C.to_atelier_garment(spec, curve_steps=curve_steps, sew_steps=sew_steps)
        return (json.dumps(garment, ensure_ascii=False),)


class GarmentSpecPreviewNode:
    """显示板片/缝纫统计，便于确认生成结果。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"atelier_garment_json": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "preview"
    CATEGORY = "GarmentPattern"

    def preview(self, atelier_garment_json):
        g = json.loads(atelier_garment_json)
        pieces, sewings = g.get("pieces", []), g.get("sewings", [])
        pts = sum(len(p.get("outline", [])) for p in pieces)
        lines = [
            "板片: %d" % len(pieces),
            "轮廓点: %d" % pts,
            "缝纫线: %d" % len(sewings),
            "---- pieces ----",
        ]
        for p in pieces:
            lines.append("  #%d %s (%d pts)" % (p["id"], p.get("name", "?"), len(p.get("outline", []))))
        return ("\n".join(lines),)


class GarmentSetupAipparelEnvNode:
    """在 RunningHub/ComfyUI 里一键准备 AIpparel 隔离环境（首次运行，需联网）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"force_reinstall": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "setup"
    CATEGORY = "GarmentPattern"

    def setup(self, force_reinstall=False):
        from setup_aipparel_env import setup
        return (setup(force=force_reinstall),)


class GarmentSaveNode:
    """把 json 存到 ComfyUI/output/garment_pattern/ 供下载。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {"multiline": True}),
                "filename": ("STRING", {"default": "garment"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "GarmentPattern"

    def save(self, json_text, filename="garment"):
        data = json.loads(json_text)
        path = _save_json(data, filename)
        return (path,)


NODE_CLASS_MAPPINGS = {
    "GarmentGenerate": GarmentGenerateNode,
    "GarmentSpecToPattern": GarmentSpecToPatternNode,
    "GarmentSpecPreview": GarmentSpecPreviewNode,
    "GarmentSave": GarmentSaveNode,
    "GarmentSetupAipparelEnv": GarmentSetupAipparelEnvNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GarmentGenerate": "AI 生成服装规格 (ChatGarment / AIpparel)",
    "GarmentSpecToPattern": "规格 JSON → 板片/缝纫",
    "GarmentSpecPreview": "板片缝纫预览",
    "GarmentSave": "保存服装 JSON",
    "GarmentSetupAipparelEnv": "准备 AIpparel 云端环境 (首次)",
}
