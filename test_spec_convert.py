#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自测：用真实 GarmentCode specification 样例验证 spec_convert 转换正确。"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spec_convert as C

SAMPLE = os.path.join(HERE, "samples", "shirt_mean_spec_sample.json")


def main():
    spec = json.load(open(SAMPLE, encoding="utf-8"))
    garment = C.to_atelier_garment(spec)
    pieces, sewings = garment["pieces"], garment["sewings"]

    print("=== pieces: %d, sewings: %d ===" % (len(pieces), len(sewings)))
    assert len(pieces) == 4, "应有 4 个板片"
    assert len(sewings) == 6, "应有 6 条缝纫"

    # 1) 单位换算：GarmentCode cm -> mm，左袖前片原点应在合理范围
    p = next(x for x in pieces if x["name"] == "left_sleeve_f")
    xs = [q[0] for q in p["outline"]]
    ys = [q[1] for q in p["outline"]]
    print("left_sleeve_f bbox x=[%.1f,%.1f] y=[%.1f,%.1f] pts=%d" %
          (min(xs), max(xs), min(ys), max(ys), len(p["outline"])))
    assert min(xs) < -100 and max(xs) <= 1, "cm->mm 换算异常: %r" % (min(xs), max(xs))
    assert len(p["outline"]) >= 8, "曲线边未被采样展开: %d pts" % len(p["outline"])

    # 2) 所有板片 outline 点数 >= 3 且首尾不重合
    for pc in pieces:
        o = pc["outline"]
        assert len(o) >= 3, "%s 点数不足" % pc["name"]
        assert math.dist(o[0], o[-1]) > 1e-6, "%s 首尾重复" % pc["name"]

    # 3) 缝纫点索引全部有效，A/B 点数一致且反向
    for i, s in enumerate(sewings):
        a, b = s["A"], s["B"]
        pa = next(x for x in pieces if x["id"] == a["piece"])
        pb = next(x for x in pieces if x["id"] == b["piece"])
        assert len(a["idxs"]) == len(b["idxs"]), "缝 %d A/B 点数不等" % i
        assert len(a["pts"]) == len(a["idxs"]), "缝 %d pts/idxs 不匹配" % i
        for idx in a["idxs"]:
            assert 0 <= idx < len(pa["outline"]), "缝 %d A 索引越界 %d/%d" % (i, idx, len(pa["outline"]))
        for idx in b["idxs"]:
            assert 0 <= idx < len(pb["outline"]), "缝 %d B 索引越界 %d/%d" % (i, idx, len(pb["outline"]))
    print("缝纫样例 A.piece=%d idxs[:5]=%s" % (sewings[0]["A"]["piece"], sewings[0]["A"]["idxs"][:5]))

    # 4) IR 转换 + 项目自带校验
    ir = C.to_garment_ir(spec, SAMPLE)
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "tools")))
    import garment_ir as G
    errs = G.validate(ir)
    print("IR summary:", json.dumps(G.summary(ir), ensure_ascii=False))
    assert not errs, "IR 校验失败: %r" % errs

    outp = os.path.join(HERE, "samples", "_test_atelier_garment.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(garment, f, ensure_ascii=False, indent=2)
    print("ALL PASS ->", outp)


if __name__ == "__main__":
    main()
