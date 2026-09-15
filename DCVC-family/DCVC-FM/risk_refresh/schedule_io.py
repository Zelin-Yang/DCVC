#!/usr/bin/env python3
"""
schedule_io.py — P0 schedule 的生成 / 校验 / 逐帧展开

用法:
  # 生成固定策略 schedule (B0/B1/B3 复刻 stock runner)
  python3 schedule_io.py generate --kind B0 --sequence_id videoSRC05_1920x1080_25 \
      --src_path /abs/path.yuv --width 1920 --height 1080 --frame_num 96 \
      --q 21 --out schedules/B0_videoSRC05_q21.json

  # 校验已有 schedule 并打印动作统计
  python3 schedule_io.py validate schedules/B0_videoSRC05_q21.json
"""

import argparse
import json
import sys

SCHEMA_VERSION = 1
MODES = ("I", "P", "P_RESET")


def make_fixed_schedule(kind, sequence_id, src_path, width, height, frame_num, q,
                        schedule_id=None):
    """按手册 Step 6 复刻 stock runner 的固定策略:
      B0: frame 0 = I, 其余 P
      B1: frame 0 = I, frame_idx % 32 == 1 -> P_RESET, 其余 P   (stock reset_interval=32)
      B3: frame 0/32/64 = I, 其余 P                            (stock force_intra_period=32)
    """
    actions = []
    for t in range(frame_num):
        if kind == "B0":
            mode = "I" if t == 0 else "P"
            reason = "initial_intra" if t == 0 else "fixed"
        elif kind == "B1":
            if t == 0:
                mode, reason = "I", "initial_intra"
            elif t % 32 == 1:
                mode, reason = "P_RESET", "fixed_reset"
            else:
                mode, reason = "P", "fixed"
        elif kind == "B3":
            mode = "I" if t % 32 == 0 else "P"
            reason = ("periodic_intra" if mode == "I" else "fixed")
            if t == 0:
                reason = "initial_intra"
        else:
            raise ValueError(f"unknown kind: {kind}")
        actions.append({"frame_idx": t, "mode": mode, "q_index": q, "reason": reason})

    return {
        "schema_version": SCHEMA_VERSION,
        "schedule_id": schedule_id or f"{kind}_{sequence_id}_q{q}",
        "lookahead": 0,
        "sequence": {
            "sequence_id": sequence_id,
            "src_path": src_path,
            "width": width,
            "height": height,
            "frame_num": frame_num,
        },
        "actions": actions,
    }


def validate(schedule):
    """校验 schedule 结构, 返回逐帧动作列表 [{mode, q_index, reason}, ...]。
    任何违规直接 AssertionError —— 拒绝带病的 schedule 进 runner。"""
    assert schedule["schema_version"] == SCHEMA_VERSION, "schema_version"
    assert schedule["lookahead"] == 0, "lookahead must be 0 (zero-look-ahead)"
    seq = schedule["sequence"]
    frame_num = seq["frame_num"]
    assert seq["width"] % 2 == 0 and seq["height"] % 2 == 0, "YUV420 needs even geometry"
    actions = schedule["actions"]

    table = [None] * frame_num
    for a in actions:
        t = a["frame_idx"]
        assert 0 <= t < frame_num, f"frame_idx {t} out of range"
        assert table[t] is None, f"duplicate action at frame {t}"
        assert a["mode"] in MODES, f"bad mode {a['mode']}"
        assert 0 <= int(a["q_index"]) <= 63, f"bad q_index {a['q_index']}"
        table[t] = {"mode": a["mode"], "q_index": int(a["q_index"]),
                    "reason": a["reason"]}

    assert table[0] is not None and table[0]["mode"] == "I", "frame 0 must be I"
    missing = [t for t, x in enumerate(table) if x is None]
    assert not missing, f"missing actions at frames {missing[:10]}..."
    return table


def summarize(schedule):
    table = validate(schedule)
    n_i = sum(1 for x in table if x["mode"] == "I")
    n_r = sum(1 for x in table if x["mode"] == "P_RESET")
    n_p = sum(1 for x in table if x["mode"] == "P")
    qs = sorted({x["q_index"] for x in table})
    print(f"schedule_id : {schedule['schedule_id']}")
    print(f"sequence    : {schedule['sequence']['sequence_id']} "
          f"{schedule['sequence']['width']}x{schedule['sequence']['height']} "
          f"{schedule['sequence']['frame_num']} frames")
    print(f"actions     : I={n_i}  P_RESET={n_r}  P={n_p}   q_indices={qs}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--kind", required=True, choices=["B0", "B1", "B3"])
    g.add_argument("--sequence_id", required=True)
    g.add_argument("--src_path", required=True)
    g.add_argument("--width", type=int, required=True)
    g.add_argument("--height", type=int, required=True)
    g.add_argument("--frame_num", type=int, required=True)
    g.add_argument("--q", type=int, required=True)
    g.add_argument("--codec_commit", default="")
    g.add_argument("--out", required=True)

    v = sub.add_parser("validate")
    v.add_argument("path")

    args = ap.parse_args()

    if args.cmd == "generate":
        sch = make_fixed_schedule(args.kind, args.sequence_id, args.src_path,
                                  args.width, args.height, args.frame_num, args.q)
        if args.codec_commit:
            sch["codec_commit"] = args.codec_commit
        validate(sch)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(sch, f, indent=1)
        print(f"written: {args.out}")
        summarize(sch)
    else:
        with open(args.path, encoding="utf-8") as f:
            summarize(json.load(f))


if __name__ == "__main__":
    main()