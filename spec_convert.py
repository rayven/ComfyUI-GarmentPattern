#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spec_convert.py — GarmentCode / ChatGarment 的 specification.json 转换层。

把 AI 生成的服装规格 JSON（pattern.panels + pattern.stitches）转成两种产物：
  1) Atelier garment json:  {"pieces":[{"name","id","outline"}], "sewings":[...]}
     —— 直接喂给项目里 tools/atelier_rebuild_lite.py 等重建工具
  2) garment-ir/1.0:        供 out/garment_viewer.html 可视化 / 存档比对

关键点（都基于真实样例 shirt_mean_specification.json 实测）：
  - 坐标单位：GarmentCode properties.units_in_meter=100 → 1 unit = 1 cm。
    Atelier / IR 用 mm，所以所有点 ×10。
  - panel.vertices 只是角点；edges 上的 curvature(quadratic/cubic) 是 Bezier，
    必须采样展开成折线，否则曲线边会变成直线，板片形状失真。
  - stitches 是「整条边对整条边」，Atelier 要的是 outline 上的点索引 + 坐标，
    所以要把 edge 映射到采样后的索引区间，并把两条边重采样到相同点数。

本模块只用标准库，不依赖 torch / ComfyUI，可在本机独立测试。
"""
import json
import math
import os
import sys

MM_PER_UNIT = 10.0  # GarmentCode cm -> mm


# ---------------------------------------------------------------- 曲线采样
def _bezier(p0, p1, ctrl, n):
    """采样 Bezier 曲线。p0/p1 为端点全局坐标，ctrl 为若干控制点全局坐标。"""
    pts_ctrl = [p0] + list(ctrl) + [p1]
    degree = len(pts_ctrl) - 1
    if degree < 2:
        return [list(p0), list(p1)]
    out = []
    for i in range(n + 1):
        t = i / n
        x = y = 0.0
        for k, p in enumerate(pts_ctrl):
            # Bernstein 基函数
            c = math.comb(degree, k) * ((1 - t) ** (degree - k)) * (t ** k)
            x += c * p[0]
            y += c * p[1]
        out.append([x, y])
    return out


def _edge_points(p0, p1, edge, curve_steps):
    """把一条 edge 展开成折线点列（含起点，不含终点，便于拼接）。"""
    curvature = edge.get("curvature")
    if not curvature or curvature.get("type") not in ("quadratic", "cubic"):
        return [list(p0)]
    params = curvature.get("params") or []
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    # GarmentCode curvature_coords=relative: 控制点 (t,s) 相对边向量
    # ctrl = p0 + t*(p1-p0) + s*perp(p1-p0), perp(dx,dy)=(-dy,dx)
    ctrl = [(p0[0] + t * dx - s * dy, p0[1] + t * dy + s * dx) for t, s in params]
    pts = _bezier(p0, p1, ctrl, curve_steps)
    return pts[:-1]  # 去掉终点，由下一条边的起点补上


def sample_panel(panel, curve_steps=12, mm_per_unit=MM_PER_UNIT):
    """把一个 panel 的 vertices+edges 展开成闭合 outline(mm) + edge→索引区间映射。

    返回:
      outline: [[x_mm,y_mm], ...]  首尾不重复
      edge_span: {edge_index: (start_idx, end_idx_inclusive)}  基于 outline 索引
    """
    verts = panel.get("vertices") or []
    edges = panel.get("edges") or []
    if len(verts) < 3 or not edges:
        return [], {}

    outline_local = []
    edge_start = {}
    for ei, edge in enumerate(edges):
        a, b = edge.get("endpoints", [0, 0])
        p0, p1 = verts[a], verts[b]
        edge_start[ei] = len(outline_local)
        seg = _edge_points(p0, p1, edge, curve_steps)
        if not seg:
            seg = [list(p0)]
        outline_local.extend(seg)  # 每条边含起点、不含终点；终点由下一条边起点表示

    total = len(outline_local)
    edge_span = {}
    for ei in range(len(edges)):
        end = edge_start.get((ei + 1) % len(edges), 0)  # 最后一条边终点回到 0
        edge_span[ei] = (edge_start[ei], end)
    outline = [[round(x * mm_per_unit, 3), round(y * mm_per_unit, 3)] for x, y in outline_local]
    return outline, edge_span


# ---------------------------------------------------------------- 缝纫映射
def _resample_indices(span, target_n, n_points):
    """把一个 edge 的索引区间均匀重采样成 target_n 个索引（含两端），支持跨 0 回绕。"""
    a, b = span
    seq = []
    cur = a
    while True:
        seq.append(cur)
        if cur == b:
            break
        cur = (cur + 1) % n_points
        if len(seq) > n_points:  # 防御死循环
            break
    if len(seq) == 0:
        return []
    if len(seq) == 1:
        return [seq[0]] * target_n
    if target_n == 1:
        return [seq[0]]
    return [seq[round(i * (len(seq) - 1) / (target_n - 1))] for i in range(target_n)]


def build_sewings(spec, panel_index, spans_by_panel, points_by_panel, sew_steps=12):
    """把 pattern.stitches 转成 Atelier sewing 列表。"""
    stitches = (spec.get("pattern") or {}).get("stitches") or []
    sewings = []
    for si, stitch in enumerate(stitches):
        if not isinstance(stitch, (list, tuple)) or len(stitch) != 2:
            continue
        ends = []
        ok = True
        for side in stitch:
            name = side.get("panel")
            ei = side.get("edge")
            if name not in spans_by_panel or ei not in spans_by_panel[name]:
                ok = False
                break
            ends.append((name, spans_by_panel[name][ei]))
        if not ok:
            continue
        (na, span_a), (nb, span_b) = ends
        n = min(sew_steps, max(2, min(len(points_by_panel[na]), len(points_by_panel[nb]))))
        idx_a = _resample_indices(span_a, n, len(points_by_panel[na]))
        idx_b = _resample_indices(span_b, n, len(points_by_panel[nb]))
        idx_b = list(reversed(idx_b))  # 两条边反向缝合
        pts_a = [points_by_panel[na][i] for i in idx_a]
        pts_b = [points_by_panel[nb][i] for i in idx_b]
        sewings.append({
            "A": {"piece": panel_index[na], "idxs": idx_a, "pts": pts_a},
            "B": {"piece": panel_index[nb], "idxs": idx_b, "pts": pts_b},
        })
    return sewings


# ---------------------------------------------------------------- 主转换
def to_atelier_garment(spec, curve_steps=12, sew_steps=12):
    """specification.json (dict) -> Atelier garment json dict。"""
    pattern = spec.get("pattern") or {}
    panels = pattern.get("panels") or {}
    props = spec.get("properties") or {}
    uim = props.get("units_in_meter", 100)
    mm_per_unit = 1000.0 / float(uim) if uim else MM_PER_UNIT

    order = pattern.get("panel_order") or list(panels.keys())
    order = [n for n in order if n in panels]
    panel_index = {name: i for i, name in enumerate(order)}

    pieces, spans_by_panel, points_by_panel = [], {}, {}
    for name in order:
        outline, spans = sample_panel(panels[name], curve_steps, mm_per_unit)
        pieces.append({"name": name, "id": panel_index[name], "outline": outline})
        spans_by_panel[name] = spans
        points_by_panel[name] = outline

    sewings = build_sewings(spec, panel_index, spans_by_panel, points_by_panel, sew_steps)
    return {"pieces": pieces, "sewings": sewings}


def to_garment_ir(spec, source_file="", curve_steps=12):
    """specification.json (dict) -> garment-ir/1.0 dict（复用项目 schema）。"""
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tools")))
    try:
        import garment_ir as G
    except Exception:
        G = None
    atelier = to_atelier_garment(spec, curve_steps)
    ir = {
        "schema": "garment-ir/1.0",
        "meta": {
            "source_format": "garmentcode-specification",
            "source_file": source_file or "",
            "units": {"length": "mm", "area": "mm^2", "angle": "deg"},
            "notes": ["converted from ChatGarment/GarmentCode specification.json"],
        },
        "patterns": [
            {"id": str(p["id"]), "name": p["name"], "outline": p["outline"]}
            for p in atelier["pieces"]
        ],
        "sewing": atelier["sewings"],
        "fabrics": [],
        "textures": [],
        "meshes": {"vertex_count": 0, "face_count": 0, "piece_count": len(atelier["pieces"])},
        "coverage": {
            "patterns": {"status": "full", "detail": "2D outline from panel vertices+curvature"},
            "sewing": {"status": "full", "detail": "edge-to-edge stitches resampled"},
            "fabrics": {"status": "none", "detail": ""},
            "textures": {"status": "none", "detail": ""},
        },
    }
    if G:
        ir = G.finalize(ir)
    return ir


# ---------------------------------------------------------------- CLI
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("spec_json")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--ir", action="store_true", help="输出 garment-ir 而非 atelier garment json")
    ap.add_argument("--curve-steps", type=int, default=12)
    a = ap.parse_args()

    with open(a.spec_json, "r", encoding="utf-8") as f:
        spec = json.load(f)
    out = to_garment_ir(spec, a.spec_json, a.curve_steps) if a.ir else to_atelier_garment(spec, a.curve_steps)
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print("wrote", a.out)
    else:
        print(txt[:4000])


if __name__ == "__main__":
    main()
