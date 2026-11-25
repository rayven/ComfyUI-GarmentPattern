#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aipparel_runner.py — AIpparel 官方推理链的 ComfyUI/RunningHub 包装器。

不是重新实现模型：同官方 scripts/inference.py 的调用顺序逐行对齐，只去掉 hydra，
并把输入改成 ComfyUI 节点传过来的图片/文字、输出改成可供节点读取的 manifest。

官方代码来源（MIT）：
  https://github.com/georgeNakayama/AIpparel-Code  commit c08e089
  已原样打包在本插件 aipparel/ 目录，仅把绘图依赖改为按需导入。

调用示例：
  python aipparel_runner.py --weights <aipparel_pretrained.safetensors> \
      --image /tmp/input.png --out-dir /tmp/run --manifest /tmp/run/result.json
"""
import argparse
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(HERE, "aipparel")
sys.path.insert(0, VENDOR_DIR)

# HuggingFace cache 默认放在插件目录内，云端多次运行可复用，不用每次重下 LLaVA/CLIP
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HERE, ".hf_cache", "hub"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from functools import partial  # noqa: E402

from models.aipparel_model import AIpparelForCausalLM  # noqa: E402
from models.llava import conversation as conversation_lib  # noqa: E402
from data.data_wrappers.collate_fns import collate_fn  # noqa: E402
from data.datasets.inference_dataset import InferenceDataset  # noqa: E402
from data.datasets.panel_configs import (  # noqa: E402
    StandardizeConfig, StatsConfig,
)
from data.datasets.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN  # noqa: E402
from data.garment_tokenizers.gcd_garment_tokenizer import GCDGarmentTokenizer  # noqa: E402
from trainers.utils import dict_to_cuda, dict_to_cpu, dict_to_dtype  # noqa: E402


# 与官方 configs/aipparel_inference.yaml 完全一致，勿随意改动（会改变去归一化结果）
GT_STATS = StandardizeConfig(
    rotations=StatsConfig(scale=[1, 1, 1, 1], shift=[0, 0, 0, 0]),
    translations=StatsConfig(
        scale=[26.06867645, 32.42920198, 22.29905009],
        shift=[-1.25378371e-02, 1.13507532e+02, 2.63046369e+00],
    ),
    vertices=StatsConfig(
        scale=[24.4920733, 26.60402835],
        shift=[8.44428116, 16.84081321],
    ),
)

# 与官方 configs/model/aipparel_model.yaml 一致
MODEL_KWARGS = dict(
    use_mm_start_end=True,
    edge_loss_weight=0.1,
    pos_embed=True,
    num_freq=0,
    pos_embed_type="sinusoidal",
    verts_bounds=[-4, -4, 4, 4],
    transf_bounds=[-4, -4, -4, -1, -1, -1, -1, 4, 4, 4, 1, 1, 1, 1],
    denormalize_for_loss=False,
    num_regression_layers=2,
    discretize=True,
    bin_num=256,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_state_dict(weights_path):
    """官方 .pth 与用户转换的 .safetensors 均支持；只映射到 CPU，避免占显存。"""
    if weights_path.lower().endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise RuntimeError("加载 safetensors 需要 pip install safetensors") from e
        return load_file(weights_path, device="cpu")
    return torch.load(weights_path, map_location="cpu")


def find_specs(out_dir):
    found = []
    for root, _, files in os.walk(out_dir):
        for name in files:
            if name.endswith("_specification.json"):
                found.append(os.path.join(root, name))
    return sorted(found)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    inference_json = os.path.join(args.out_dir, "_inference_input.json")
    entries = []
    if args.image and os.path.isfile(args.image):
        entries.append({"type": "image", "inputs": {"image_path": args.image}})
    if (not entries or args.include_prompt) and args.prompt:
        entries.append({"type": "description", "inputs": {"description": args.prompt}})
    if not entries:
        raise RuntimeError("未提供图片，也未提供文字描述，无法推理")
    with open(inference_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, cache_dir=args.cache_dir or None,
        model_max_length=2100, padding_side="right", use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    tokenizer_cls = GCDGarmentTokenizer(
        standardize=GT_STATS, random_tag=True, num_tags=108
    )
    dataset = InferenceDataset(
        inference_json=inference_json,
        vision_tower=args.vision_tower,
        image_size=224,
        garment_tokenizer=tokenizer_cls,
        panel_classification=os.path.join(VENDOR_DIR, "assets", "data_configs",
                                          "panel_classes_garmentcodedata.json"),
    )
    all_new_tokens = dataset.get_all_token_names()
    tokenizer.add_tokens(all_new_tokens)
    token_name2idx = {}
    for token in all_new_tokens:
        token_name2idx[token] = tokenizer(token, add_special_tokens=False).input_ids[0]
    dataset.set_token_indices(token_name2idx)

    dtype = DTYPES[args.precision]
    model = AIpparelForCausalLM.from_pretrained(
        args.version,
        cache_dir=args.cache_dir or None,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        panel_edge_indices=dataset.panel_edge_type_indices,
        gt_stats=dataset.gt_stats,
        **MODEL_KWARGS,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=dtype, device="cuda")
    model.resize_token_embeddings(len(tokenizer))
    for p in model.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]

    state_dict = load_state_dict(args.weights)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("AIpparel weights loaded: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    del state_dict
    model = model.to("cuda")
    model.eval()

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
        collate_fn=partial(
            collate_fn, tokenizer=tokenizer, conv_type="llava_v1",
            use_mm_start_end=MODEL_KWARGS["use_mm_start_end"], local_rank=0,
            generation_only=True,
        ),
    )

    specs, output_texts = [], []
    for i, input_dict in enumerate(loader):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)
        input_dict = dict_to_dtype(input_dict, dtype, [
            "images_clip", "param_targets", "param_target_endpoints",
            "param_target_transformations", "questions_pattern_endpoints",
            "questions_pattern_transformations",
        ])
        output_dict = model.evaluate(
            input_dict["images_clip"], input_dict["question_ids"],
            input_dict["question_attention_masks"],
            endpoints=input_dict["questions_pattern_endpoints"],
            endpoints_mask=input_dict["questions_pattern_endpoints_mask"],
            transformations=input_dict["questions_pattern_transformations"],
            transformations_mask=input_dict["questions_pattern_transformations_mask"],
            max_new_tokens=args.max_new_tokens,
        )
        output_dict = dict_to_dtype(dict_to_cpu(output_dict), torch.float32)
        output_dict["input_mask"] = (
            torch.arange(output_dict["output_ids"].shape[1]).reshape(1, -1)
            >= input_dict["question_ids"].shape[1]
        )
        output_text, patterns, error_type = dataset.decode(output_dict, tokenizer)
        sample_dir = os.path.join(args.out_dir, "sample_%d" % i)
        os.makedirs(sample_dir, exist_ok=True)
        patterns.serialize(sample_dir, spec_only=False, with_3d=False, with_text=False,
                           view_ids=False, to_subfolder=False, tag="_pred")
        specs.extend(find_specs(sample_dir))
        output_texts.append(output_text)

    best = None
    for path in specs:
        try:
            spec = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            continue
        if (spec.get("pattern") or {}).get("panels"):
            best = spec
            break
    if best is None:
        raise RuntimeError("推理完成但没有有效板片（pattern.panels 为空）")

    return {
        "specification": best,
        "spec_files": specs,
        "output_text": "\n---\n".join(output_texts),
        "weights": args.weights,
        "precision": args.precision,
        "transformers_version": transformers.__version__,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="aipparel_pretrained.pth/.safetensors")
    ap.add_argument("--image", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--include-prompt", action="store_true",
                    help="图片模式时额外生成一条纯文字样本，取第一个有效结果")
    ap.add_argument("--version", default="liuhaotian/llava-v1.5-7b",
                    help="LLaVA-1.5-7B 的 HF id 或本地目录（提供 config/tokenizer）")
    ap.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    ap.add_argument("--precision", choices=list(DTYPES.keys()), default="bf16")
    ap.add_argument("--cache-dir", default="", help="HF 缓存目录，默认插件目录/.hf_cache")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", required=True, help="结果 JSON 写入路径")
    ap.add_argument("--max-new-tokens", type=int, default=2100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = args.manifest
    try:
        result = run(args)
        result["error"] = None
    except Exception as e:
        traceback.print_exc()
        result = {"error": "%s: %s" % (type(e).__name__, e), "specification": None}
    os.makedirs(os.path.dirname(os.path.abspath(manifest)), exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print("RESULT_JSON:" + manifest)
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
