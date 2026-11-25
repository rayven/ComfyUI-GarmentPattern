# RunningHub 上使用 ComfyUI-GarmentPattern：当前结论

## 结论先说

我撤回上一版“在 RunningHub 工作流编辑器用 Git URL 安装自定义节点”的说法。
那是我按本地 ComfyUI / ComfyUI-Manager 的通用方式推断的，**不是 RunningHub 实测，也没有找到官方依据**。

截至 2026-11-26 我核查到的证据：

1. 你当前打开的 RunningHub RHTV 画布页面（`/projects/canvas/me`）导航栏是：
   首页 / 项目 / 资产库 / 技能 / 社区 / MCP / **插件**。
   其中“插件”页面只有 Blender、Maya、Figma、Premiere 客户端插件下载，**没有 ComfyUI Manager、没有 Install via Git URL、没有自定义节点管理**。
2. RunningHub 官方 API 文档只写：模型库收藏/上传私有模型、工作流 API、原生 ComfyUI 代理地址；
   我没有在这些官方文档里找到“自助安装任意 ComfyUI 自定义节点”的说明。
3. 第三方导航站提到 RunningHub 有“节点商店 / Python、JS 自定义节点 / 开发者上传节点”，
   但都没有给出用户侧自助安装 Git 仓库的入口，且不是官方来源，不能当依据。

所以：**我不能确认 RunningHub 提供你想要的自定义节点安装入口；当前界面证据倾向于它没有对普通用户开放。**

---

## 现在该做什么

### 1. 直接问 RunningHub 官方（最可靠）

用 RunningHub 平台内客服/反馈，把问题原样发过去：

> 是否支持在云端 ComfyUI 工作流里安装第三方自定义节点？
> 若支持，入口在哪里？是否支持从公开 GitHub 仓库 URL 安装或上传节点包？
> 是否支持上传 25.4 GiB 的私有 safetensors checkpoint 并在工作流中作为模型权重使用？
> 若不支持用户自助安装，节点开发者提交/审核的流程是什么？

在得到明确答复前，不要再按我上一版说的菜单去找，也不要把 `REGISTRY_ACCESS_TOKEN` 当成 RunningHub 凭证。

### 2. Comfy Registry 这条路现在只能算“可能有用”

`REGISTRY_ACCESS_TOKEN` 只用于把插件发布到 Comfy Registry / ComfyUI-Manager 索引。
如果 RunningHub 的节点库从 Registry 同步，发布后有可能出现；但我没有找到证据证明它一定会同步。
补齐文件仍然值得做，但不是保证能上 RunningHub。

需要补的文件在同级目录：
`pyproject.toml`（PublisherId 需替换）、`.comfyignore`、`.github/workflows/publish_action.yml`、`README.md`。

### 3. 如果 RunningHub 不支持自定义节点，实际可行的替代方案

- 在支持自定义节点的云 GPU 环境（RunPod / AutoDL / 本地有显卡的机器）跑这个插件；
- 把 AIpparel 推理封装成独立 HTTP 服务，再通过 RunningHub 工作流里的可用 HTTP 节点调用（前提是平台提供该节点）；
- 等 RunningHub 官方把 GarmentPattern 节点纳入平台节点库后再在它的工作流里使用。

---

## 真实推理的前置风险（不变）

即使节点最终能装进去，仍有三个硬门槛：

- 25.4 GiB 权重能否上传/被识别为 checkpoint；
- 是否允许安装依赖、创建 venv、出站下载 LLaVA-1.5-7B（约 13GB）与 CLIP（约 0.6GB）；
- 是否允许在运行时写插件目录和 output 目录。

报错要看 `output/garment_pattern/aipparel_runs/<时间戳>/result.json`。
