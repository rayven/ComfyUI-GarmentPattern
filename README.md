# ComfyUI-GarmentPattern（RunningHub 版）

图/文 → 服装 `specification.json` → Style3D Atelier 可用的板片轮廓 + 缝合线 JSON。

本插件支持两种生成后端：

| 后端 | 说明 | 适用 |
|---|---|---|
| `mock` | 内置样例，零依赖，验证整条链路 | 本机 / 云端首次联调 |
| `aipparel` | **AIpparel 官方推理链**（CVPR 2025 Highlight） | **RunningHub 云端 GPU** |
| `http` / `local` | ChatGarment + GarmentCodeRC 展开 | 自建云 GPU / 本机大显存 |

---

## 关键事实（已实测/核实，别踩旧坑）

1. **AIpparel 官方代码存在**：`github.com/georgeNakayama/AIpparel-Code`（MIT，commit `c08e089`）。
   之前记录的「仓库里只有网页、没有代码」是查错了仓库（`AIpparel`，少一个 `-Code`），已纠正。
2. 官方代码已按 MIT 协议**原样打包在插件 `aipparel/` 目录**，只把绘图用的
   `cairosvg` / `matplotlib` 改成按需导入，不改变模型与解码逻辑。
3. 你下载的两个权重文件互为同一份权重的两种格式：
   - `aipparel_pretrained.pth` 27,226,710,763 B
   - `aipparel_pretrained.safetensors` 27,226,699,828 B
   - 解析头部确认 **339 个张量，键名集合完全一致**；架构是 LLaVA-1.5-7B 基座
     （32 层 / hidden 4096 / vocab 32124）+ 服装专用头（`transformation_fc`、
     `line_curve_fc`、`transf_proj`、`vertex_proj`）。两者都能推理，云端优先传 safetensors。
4. **官方权重不含视觉编码器和 mm-projector**（HuggingFace 官方说明）。首次推理会自动下载
   `liuhaotian/llava-v1.5-7b`（约 13 GB）和 `openai/clip-vit-large-patch14`（约 0.6 GB），
   缓存放插件目录 `.hf_cache`，后续运行复用。
5. AIpparel 直接输出真正的 2D 板片规格（`pattern.panels` / `pattern.stitches`），
   **不需要** ChatGarment 那一步 GarmentCodeRC 展开。

---

## RunningHub 部署（前置条件尚未确认）

### 1. 安装插件：先问官方，不要按旧步骤找菜单

仓库地址：`https://github.com/rayven/ComfyUI-GarmentPattern`（公开）。

**重要更正（2026-11-26）**：之前写的“在 RunningHub 里用 ComfyUI Manager 的
Install via Git URL 安装”是按本地 ComfyUI 推断的，不是 RunningHub 实测，也没有官方依据。
当前 RHTV 画布的“插件”页只有 Blender/Maya/Figma/Premiere 客户端插件；
官方文档只写模型上传、工作流 API 和原生 ComfyUI 代理，未写自助安装任意自定义节点。
因此请先用平台内客服确认：是否支持第三方 ComfyUI 自定义节点、入口/提交审核流程是什么。
详见同目录 `RUNNINGHUB.md`。

`REGISTRY_ACCESS_TOKEN` 只是 Comfy Registry 的发布密钥，**不是** RunningHub 登录密钥，
也不要填进工作流节点。补齐 Registry 元数据是必要准备，但不能保证 RunningHub 会同步该节点。

### 2. 上传模型权重

1. RunningHub 左侧 **模型库 → 上传模型**，类型选 **Checkpoint**，文件选
   `aipparel_pretrained.safetensors`（约 25.4 GiB；推荐 safetensors，平台若限制大小再试 `.pth`）。
2. 上传后收藏该模型，刷新工作流页面，生成节点的 `weights` 下拉才会出现它。
3. 如果平台模型库不收这个类型/体积：把文件放到云端可访问路径，填到
   `weights_path` 输入框（绝对路径优先于下拉）。

### 3. 准备隔离环境（首次，一次即可）

在画布里放一个 **`准备 AIpparel 云端环境 (首次)`** 节点并运行一次。它会：

- 在插件目录创建 `.venv_aipparel`（`--system-site-packages`，继承 ComfyUI 的 torch/CUDA）；
- 在其中安装 `transformers==4.31.0` 等 AIpparel 锁定版本，**不污染 RunningHub 全局环境**；
- 需要联网。若云端禁止 venv/pip，需联系平台或用已装好依赖的镜像。

### 4. 配置生成节点

`AI 生成服装规格 (ChatGarment / AIpparel)` 节点：

| 参数 | 值 |
|---|---|
| `mode` | `aipparel` |
| `image` | 可选，接 LoadImage 的衣服图；不接则纯文字 |
| `prompt` | 例如 `A dress with a square neckline and a pleated skirt` |
| `weights` | 第 2 步上传的模型名 |
| `python_executable` | 留空即可：若第 3 步建好了 `.venv_aipparel` 会自动使用；否则填 `.venv_aipparel/Scripts/python.exe` |
| `precision` | `bf16`（推荐）；显存紧张且报错再试 `fp16` |
| `timeout` | 默认 7200 秒（首次要下载 LLaVA/CLIP） |
| `include_prompt` | 有图片时是否再跑一条纯文字样本（默认关） |

**显存**：官方权重是 FP32（约 26 GB），加载时转 BF16 后约 13 GB，加上 LLaVA/CLIP 与
2100 token 生成，建议选 **48 GB 显存机器**；24 GB 机器可能成功但余量很小。

### 5. 工作流

```
LoadImage ─┐
           ├→ GarmentGenerate(mode=aipparel) → GarmentSpecToPattern → GarmentSpecPreview
Prompt ────┘                                      ↓
                                            GarmentSave → output/garment_pattern/*.json
```

下载的 JSON 喂给项目里的 `tools/atelier_rebuild_lite.py` 等，在 Style3D Atelier 里画板片与缝合线。
每次 AIpparel 运行的原始规格、输入图、输出文本都保存在
`output/garment_pattern/aipparel_runs/<时间戳>/`，便于排错。

---

## 验证状态（如实说明）

**已在本机静态验证：**
- `.pth` 与 `.safetensors` 头部/键名一致；确认是 AIpparel 官方权重。
- 官方仓库源码可下载、MIT 许可、依赖中没有 deepspeed/flash-attn 硬依赖。
- 转换层对 GarmentCode 规格的转换（cm→mm、Bezier 采样、缝纫点索引）之前已实测通过。
- 节点注册与 mock→转换→预览→保存链路之前已在真实 ComfyUI 环境跑通。

**未验证（按你的要求不在本机跑）：**
- **真实 AIpparel 推理与 RunningHub 云端部署**：本机显卡不够，未加载 26 GB 模型。
  云端首次运行才有真实结论；如遇依赖/显存/平台限制，错误信息会写进
  `output/garment_pattern/aipparel_runs/<时间戳>/result.json` 与工作流日志。

---

## 依赖

核心转换层零第三方依赖。`aipparel_requirements.txt` 只用于**隔离环境**，取自官方
`requirements.txt` 的推理必需子集。官方 LICENSE 见 `aipparel/LICENSE`。
