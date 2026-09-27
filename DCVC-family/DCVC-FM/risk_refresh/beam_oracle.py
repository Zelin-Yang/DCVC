#!/usr/bin/env python3
"""
beam_oracle.py — P1.4/P1.5 预算约束 beam-search oracle (协议 §8/§9) + P1-R2 protected incumbent
有限动作、预算约束的 beam oracle —— 非全局最优, 名称纪律见协议 §9.1。

机制: 15 段 x 8 帧决策网格; 每节点携带完整编码状态 (DPB/SPS/码流前缀/累计量);
扩展=真实编码 8 帧段 (BytesIO 码流); 剪枝=硬约束过滤 -> Pareto -> 分层截断;
frontier 轨迹 100% 完整解码审计 (协议 11.2)。

P1-R2 (2026-09-24, 评审要求):
  --incumbent_schedule <grid_baseline.json>  网格兼容固定基线 (非P动作必须在8的倍数帧)
  --protect_incumbent  每 depth 强制保留 incumbent 前缀 (挤占一个束位)
  incumbent 预编码 + 全约束自检 (bits/peak/计数/spacing), 违规即中止并列出违规项;
  终局输出 incumbent 独立行; frontier 最优劣于 incumbent -> ENGINE FAILURE (该次运行 invalid)。

用法:
  python3 -m risk_refresh.beam_oracle --sequence_id ... --src_path ... \
    --q_init 0 --oracle resetonly --budget_bits N --peak_budget_bits M \
    --beam_width 8 --out_dir ... [--n_i_max 1] [--n_reset_max 4] [--i_spacing 16] \
    [--incumbent_schedule B1grid.json] [--protect_incumbent]
"""

import argparse
import csv
import hashlib
import io
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.stream_helper import (
    SPSHelper, NalType, write_sps, read_header, read_sps_remaining, read_ip_remaining)
from src.utils.video_reader import YUVReader
from src.transforms.functional import ycbcr420_to_444, ycbcr444_to_420
from src.utils.metrics import calc_psnr

from risk_refresh.schedule_runner import (
    init_models, calc_distortion, recon_hash, INDEX_MAP, sha256_file)
from risk_refresh.schedule_io import validate as validate_schedule

SEG_LEN = 8
FRAME_NUM = 120
EPS = 0.005


# ────────────────── 帧加载 ──────────────────

def load_frames(src_path, width, height, frame_num):
    """预载全部帧: ys [N,1,H,W], uvs [N,2,H/2,W/2] (uint8, 与 runner 的 y/uv 同源)"""
    reader = YUVReader(src_path, width, height)
    ys, uvs = [], []
    for _ in range(frame_num):
        y, uv = reader.read_one_frame(dst_format="420")
        ys.append(y.copy())
        uvs.append(uv.copy())
    reader.close()
    return np.stack(ys), np.stack(uvs)


# ────────────────── 段编码 ──────────────────

def encode_segment(i_net, p_net, dpb_in, spss_in, ys_np, uvs_np, t0, mode, q,
                   device, padding, src_width, src_height,
                   seg_len=SEG_LEN):
    """编码 [t0, t0+8) 段: 首帧按 mode, 其余普通 P(q)。
    返回 (seg_bytes, frame_logs, dpb_out, spss_out)。"""
    buf = io.BytesIO()
    helper = SPSHelper()
    helper.spss = [dict(s) for s in spss_in]
    dpb = {k: (v.clone() if torch.is_tensor(v) else None) for k, v in dpb_in.items()}
    logs = []
    outstanding = 0
    pl, pr, pt, pb = padding

    with torch.no_grad():
        for k in range(seg_len):
            t = t0 + k
            yuv = ycbcr420_to_444(ys_np[t], uvs_np[t])
            x = torch.from_numpy(yuv).type(torch.FloatTensor).unsqueeze(0).to(device)
            x_pad = F.pad(x, (pl, pr, pt, pb), mode="replicate")

            if k == 0 and mode == "I":
                sps = {"sps_id": -1, "height": src_height, "width": src_width,
                       "qp": q, "fa_idx": 0}
                sps_id, sps_new = helper.get_sps_id(sps)
                sps["sps_id"] = sps_id
                if sps_new:
                    outstanding += write_sps(buf, sps)
                result = i_net.encode(x_pad, q, sps_id, buf)
                dpb = {"ref_frame": result["x_hat"], "ref_feature": None,
                       "ref_mv_feature": None, "ref_y": None, "ref_mv_y": None}
                recon = result["x_hat"]
                fa_idx = 0
            else:
                fa_idx = INDEX_MAP[t % 8]
                if k == 0 and mode == "P_RESET":
                    dpb["ref_feature"] = None
                    fa_idx = 3
                sps = {"sps_id": -1, "height": src_height, "width": src_width,
                       "qp": q, "fa_idx": fa_idx}
                sps_id, sps_new = helper.get_sps_id(sps)
                sps["sps_id"] = sps_id
                if sps_new:
                    outstanding += write_sps(buf, sps)
                result = p_net.encode(x_pad, dpb, q, fa_idx, sps_id, buf)
                dpb = result["dpb"]
                recon = dpb["ref_frame"]

            recon = recon.clamp_(0, 1)
            x_hat = F.pad(recon, (-pl, -pr, -pt, -pb))
            y_true, uv_true = ys_np[t], uvs_np[t]
            yuv_rec = x_hat.squeeze(0).cpu().numpy()
            y_rec, uv_rec = ycbcr444_to_420(yuv_rec)
            psnr_y = calc_psnr(y_true[0], y_rec[0], data_range=1)
            psnr_u = calc_psnr(uv_true[0], uv_rec[0], data_range=1)
            psnr_v = calc_psnr(uv_true[1], uv_rec[1], data_range=1)
            sse_y = float(np.asarray(y_true[0] - y_rec[0], dtype=np.float64).__pow__(2).sum())
            sse_u = float(np.asarray(uv_true[0] - uv_rec[0], dtype=np.float64).__pow__(2).sum())
            sse_v = float(np.asarray(uv_true[1] - uv_rec[1], dtype=np.float64).__pow__(2).sum())
            bits = result["bit"] + outstanding * 8
            logs.append({"frame_idx": t, "mode": mode if k == 0 else "P",
                         "q_index": q, "fa_idx": fa_idx, "bits": int(bits),
                         "sse_y": sse_y, "sse_u": sse_u, "sse_v": sse_v,
                         "psnr_y": psnr_y, "psnr_u": psnr_u, "psnr_v": psnr_v,
                         "recon_sha256": recon_hash(x_hat)})
            outstanding = 0

    return buf.getvalue(), logs, dpb, helper.spss


# ────────────────── P1-R2: protected incumbent ──────────────────

def parse_grid(spec):
    """'8:8-56,1:56-77,8:80-120' -> (decisions 升序列表, {t: seg_len})。
    区域须无缝铺满 [SEG_LEN, FRAME_NUM); 首决策点必须 == SEG_LEN (seed 恒 8 帧)。"""
    dec = set()
    for zone in spec.split(","):
        step_s, rng = zone.split(":")
        lo, hi = (int(x) for x in rng.split("-"))
        t = lo
        while t < hi:
            dec.add(t)
            t += int(step_s)
    decisions = sorted(dec)
    assert decisions and decisions[0] == SEG_LEN, "grid: first decision must be frame 8"
    assert decisions[-1] < FRAME_NUM
    lens = {}
    for i, t in enumerate(decisions):
        end = decisions[i + 1] if i + 1 < len(decisions) else FRAME_NUM
        lens[t] = end - t
    return decisions, lens


def encode_incumbent(args, i_net, p_net, ys_np, uvs_np, device,
                     budget, peak_budget, padding, decisions, lens):
    """预编码网格兼容 incumbent。
    返回 (seed_node, plan, violations):
      seed_node — 段0 (I+7P) 之后的前缀节点, beam 从 t=8 起扩展它;
      plan      — {t0: (mode, q)} incumbent 在每个决策段的动作;
      violations 非空 -> 基线在 oracle 约束下不可行, 调用方必须中止。
    自检项: 网格兼容性 (非P动作仅在8倍数帧, 段内q一致), bits 总预算,
            rolling-32 峰值, N_I/N_reset 计数, I 帧间隔。
    """
    with open(args.incumbent_schedule, encoding="utf-8") as f:
        sch = json.load(f)
    table = validate_schedule(sch)
    violations = []

    dec_set = set(decisions)
    for t in range(1, FRAME_NUM):
        if table[t]["mode"] != "P" and t not in dec_set:
            violations.append(f"grid: non-P action at frame {t} not a decision point")
    plan = {}
    for t0 in decisions:
        qs = {table[t0 + k]["q_index"] for k in range(lens[t0])}
        if len(qs) != 1:
            violations.append(f"grid: segment {t0} mixed q {sorted(qs)}")
        plan[t0] = (table[t0]["mode"], table[t0]["q_index"])
    if table[0]["mode"] != "I":
        violations.append("grid: frame 0 must be I")
    elif table[0]["q_index"] != args.q_init:
        violations.append(
            f"grid: segment0 q={table[0]['q_index']} != q_init={args.q_init}")
    if violations:
        return None, None, violations

    pl, pr, pt, pb = padding
    dpb = {"ref_frame": None, "ref_feature": None, "ref_mv_feature": None,
           "ref_y": None, "ref_mv_y": None}
    spss = []
    logs, cum = [], 0
    n_i = n_reset = last_i = 0
    seed_node = None
    for t0, sl in [(0, SEG_LEN)] + [(t, lens[t]) for t in decisions]:
        m = table[t0]["mode"]
        q = table[t0]["q_index"]
        sb, lg, dpb, spss = encode_segment(i_net, p_net, dpb, spss, ys_np, uvs_np,
                                           t0, m, q, device, (pl, pr, pt, pb),
                                           args.width, args.height, seg_len=sl)
        logs.extend(lg)
        cum += sum(l["bits"] for l in lg)
        if t0 == 0:
            seed_node = Node(
                dpb=dpb, spss=spss,
                cum_bits=sum(l["bits"] for l in lg),
                cum_sse_y=sum(l["sse_y"] for l in lg),
                cum_sse_yuv=sum(6 * l["sse_y"] + l["sse_u"] + l["sse_v"] for l in lg) / 8,
                win_q=deque([l["bits"] for l in lg], maxlen=32),
                n_i=1, n_reset=0, last_i=0, last_refresh=0,
                hist=[(0, "I", args.q_init)], chunks=[sb], logs=list(lg))
        if m == "I":
            if t0 > 0 and t0 - last_i < args.i_spacing:
                violations.append(f"spacing: I at {t0} < {args.i_spacing} from {last_i}")
            n_i += 1
            last_i = t0
        if m == "P_RESET":
            n_reset += 1

    if cum > budget * (1 + EPS):
        violations.append(f"budget: incumbent {cum} > cap {int(budget * (1 + EPS))}")
    wq, peak = [], 0
    for l in logs:
        wq.append(l["bits"])
        if len(wq) > 32:
            wq.pop(0)
        peak = max(peak, sum(wq))
    if peak > peak_budget * (1 + EPS):
        violations.append(f"peak: incumbent {peak} > cap {int(peak_budget * (1 + EPS))}")
    if n_i > args.n_i_max:
        violations.append(f"count: N_I {n_i} (incl initial) > {args.n_i_max}")
    if n_reset > args.n_reset_max:
        violations.append(f"count: N_reset {n_reset} > {args.n_reset_max}")
    if violations:
        return None, None, violations
    return seed_node, plan, []


# ────────────────── beam 搜索 ──────────────────

class Node:
    __slots__ = ("dpb", "spss", "cum_bits", "cum_sse_y", "cum_sse_yuv", "win_q",
                 "n_i", "n_reset", "last_i", "last_refresh", "hist", "chunks", "logs")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def make_candidates(oracle, q_cand, t, node, args):
    cands = []
    for q in q_cand:
        if oracle in ("qonly", "joint"):
            cands.append(("P", q))
        if oracle in ("resetonly", "joint"):
            cands.append(("P", q))
            cands.append(("P_RESET", q))
        if oracle in ("ionly", "joint"):
            cands.append(("P", q))
            if (t - node.last_i >= args.i_spacing) and (node.n_i < args.n_i_max):
                cands.append(("I", q))
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def dominated(a, b):
    """a 被 b 支配: b.bits<=a.bits 且 b.sse_y<=a.sse_y 且至少一项严格。"""
    return (b.cum_bits <= a.cum_bits and b.cum_sse_y <= a.cum_sse_y
            and (b.cum_bits < a.cum_bits or b.cum_sse_y < a.cum_sse_y))


def beam_search(args, i_net, p_net, ys_np, uvs_np, device, budget, peak_budget):
    pl, pr, pt, pb = args.padding
    depths, lens = parse_grid(getattr(args, "decision_grid", "8:8-120"))
    assert all(lens[t] >= 1 for t in depths)
    q_cand = sorted({args.q_init, 0 if args.q_init != 0 else 21})

    seed_bytes, seed_logs, dpb0, spss0 = encode_segment(
        i_net, p_net, {"ref_frame": None, "ref_feature": None, "ref_mv_feature": None,
                       "ref_y": None, "ref_mv_y": None}, [], ys_np, uvs_np, 0, "I",
        args.q_init, device, (pl, pr, pt, pb), args.width, args.height)
    root = Node(dpb=dpb0, spss=spss0,
                cum_bits=sum(l["bits"] for l in seed_logs),
                cum_sse_y=sum(l["sse_y"] for l in seed_logs),
                cum_sse_yuv=sum(6 * l["sse_y"] + l["sse_u"] + l["sse_v"] for l in seed_logs) / 8,
                win_q=deque([l["bits"] for l in seed_logs], maxlen=32),
                n_i=1, n_reset=0, last_i=0, last_refresh=0,
                hist=[(0, "I", args.q_init)], chunks=[seed_bytes], logs=list(seed_logs))

    # P1-R2: incumbent 预编码 + 约束自检
    inc_node = None
    inc_plan = None
    if args.incumbent_schedule:
        inc_node, inc_plan, inc_errs = encode_incumbent(
            args, i_net, p_net, ys_np, uvs_np, device, budget, peak_budget,
            (pl, pr, pt, pb), depths, lens)
        if inc_errs:
            raise SystemExit("incumbent infeasible under oracle constraints:\n  - "
                             + "\n  - ".join(inc_errs))
        print("incumbent pre-encoded & feasible (protected)")

    beam = [root]
    prune_log = {"budget_violation": 0, "peak_violation": 0, "count_violation": 0,
                 "dominated": 0, "beam_limit": 0,
                 "incumbent_survived": 0, "incumbent_rescued": 0}

    for t in depths:
        sl = lens[t]
        expanded = []
        for node in beam:
            for mode, q in make_candidates(args.oracle, q_cand, t, node, args):
                if mode == "P_RESET" and node.n_reset >= args.n_reset_max:
                    prune_log["count_violation"] += 1
                    continue
                if mode == "I" and (node.n_i >= args.n_i_max or t - node.last_i < args.i_spacing):
                    prune_log["count_violation"] += 1
                    continue
                try:
                    seg_bytes, logs, dpb, spss = encode_segment(
                        i_net, p_net, node.dpb, node.spss, ys_np, uvs_np, t, mode, q,
                        device, (pl, pr, pt, pb), args.width, args.height,
                        seg_len=sl)
                except AssertionError:
                    prune_log["budget_violation"] += 1
                    continue
                nb = node.cum_bits + sum(l["bits"] for l in logs)
                if nb > budget * (1 + EPS):
                    prune_log["budget_violation"] += 1
                    continue
                wq = deque(node.win_q, maxlen=32)
                peak_ok = True
                for l in logs:
                    wq.append(l["bits"])
                    if sum(wq) > peak_budget * (1 + EPS):
                        peak_ok = False
                        break
                if not peak_ok:
                    prune_log["peak_violation"] += 1
                    continue
                expanded.append(Node(
                    dpb=dpb, spss=spss, cum_bits=nb,
                    cum_sse_y=node.cum_sse_y + sum(l["sse_y"] for l in logs),
                    cum_sse_yuv=node.cum_sse_yuv
                    + sum(6 * l["sse_y"] + l["sse_u"] + l["sse_v"] for l in logs) / 8,
                    win_q=wq,
                    n_i=node.n_i + (1 if mode == "I" else 0),
                    n_reset=node.n_reset + (1 if mode == "P_RESET" else 0),
                    last_i=t if mode == "I" else node.last_i,
                    last_refresh=t if mode == "P_RESET" else node.last_refresh,
                    hist=node.hist + [(t, mode, q)],
                    chunks=node.chunks + [seg_bytes],
                    logs=node.logs + logs))

        # P1-R2: incumbent 独立扩展 (前置自检已通过, 其前缀天然满足累计约束)
        inc_child = None
        if inc_node is not None:
            m, q = inc_plan[t]
            sb, lg, dpb, spss = encode_segment(
                i_net, p_net, inc_node.dpb, inc_node.spss, ys_np, uvs_np,
                t, m, q, device, (pl, pr, pt, pb), args.width, args.height,
                seg_len=sl)
            wq = deque(inc_node.win_q, maxlen=32)
            for l in lg:
                wq.append(l["bits"])
            inc_child = Node(
                dpb=dpb, spss=spss,
                cum_bits=inc_node.cum_bits + sum(l["bits"] for l in lg),
                cum_sse_y=inc_node.cum_sse_y + sum(l["sse_y"] for l in lg),
                cum_sse_yuv=inc_node.cum_sse_yuv
                + sum(6 * l["sse_y"] + l["sse_u"] + l["sse_v"] for l in lg) / 8,
                win_q=wq,
                n_i=inc_node.n_i + (1 if m == "I" else 0),
                n_reset=inc_node.n_reset + (1 if m == "P_RESET" else 0),
                last_i=t if m == "I" else inc_node.last_i,
                last_refresh=t if m == "P_RESET" else inc_node.last_refresh,
                hist=inc_node.hist + [(t, m, q)],
                chunks=inc_node.chunks + [sb],
                logs=inc_node.logs + lg)

        # Pareto 过滤
        survivors = []
        for a in expanded:
            if any(dominated(a, b) for b in expanded if b is not a):
                prune_log["dominated"] += 1
            else:
                survivors.append(a)
        # 宽度截断: 分层采样 (按 cum_bits 均匀取点, 保留两端)
        if len(survivors) > args.beam_width:
            survivors.sort(key=lambda n: n.cum_bits)
            k = args.beam_width
            idxs = sorted({round(i * (len(survivors) - 1) / max(k - 1, 1))
                           for i in range(k)} | {0, len(survivors) - 1})
            picked = [survivors[i] for i in idxs][:k]
            prune_log["beam_limit"] += len(survivors) - len(picked)
            survivors = picked
        # P1-R2: incumbent 保护 (存活检查 / 强制插回挤占束位)
        if inc_child is not None:
            alive = any(x.cum_bits == inc_child.cum_bits and x.hist == inc_child.hist
                        for x in survivors)
            if alive:
                prune_log["incumbent_survived"] += 1
            else:
                if len(survivors) >= args.beam_width:
                    survivors = survivors[:-1] + [inc_child]
                else:
                    survivors = survivors + [inc_child]
                prune_log["incumbent_rescued"] += 1
            inc_node = inc_child
        beam = survivors
        if not beam:
            raise SystemExit(f"beam 在 t={t} 处清空 —— 预算/约束过严, 检查 budget 参数")
        print(f"depth t={t:3d}: beam={len(beam)} "
              f"bits=[{min(n.cum_bits for n in beam)}..{max(n.cum_bits for n in beam)}] "
              f"prune={prune_log}")

    frontier = [a for a in beam if not any(dominated(a, b) for b in beam if b is not a)]
    return frontier, prune_log, (inc_node if args.incumbent_schedule else None)


# ────────────────── 解码审计 ──────────────────

def verify_trajectory(node, args, i_net, p_net, ys_np, uvs_np, device, out_bin):
    """完整码流 100% 解码 (协议 11.2 frontier 要求): 逐帧 hash 对照。"""
    data = b"".join(node.chunks)
    Path(out_bin).write_bytes(data)
    helper = SPSHelper()
    dpb = None
    idx = 0
    bio = io.BytesIO(data)
    with torch.no_grad():
        while idx < FRAME_NUM:
            header = read_header(bio)
            if header["nal_type"] == NalType.NAL_SPS:
                helper.add_sps_by_id(read_sps_remaining(bio, header["sps_id"]))
                continue
            if header["nal_type"] == NalType.NAL_Ps:
                pending = header["sps_ids"]
                sps_id = pending[0]
            else:
                sps_id = header["sps_id"]
            sps = helper.get_sps_by_id(sps_id)
            assert header["nal_type"] in (NalType.NAL_I, NalType.NAL_P)
            bit_stream = read_ip_remaining(bio)
            yuv = ycbcr420_to_444(ys_np[idx], uvs_np[idx])
            x = torch.from_numpy(yuv).type(torch.FloatTensor).unsqueeze(0).to(device)
            pl, pr, pt, pb = args.padding
            x_pad = F.pad(x, (pl, pr, pt, pb), mode="replicate")
            if header["nal_type"] == NalType.NAL_I:
                dec = i_net.decompress(bit_stream, sps)
                dpb = {"ref_frame": dec["x_hat"], "ref_feature": None,
                       "ref_mv_feature": None, "ref_y": None, "ref_mv_y": None}
                recon = dec["x_hat"]
            else:
                if sps["fa_idx"] == 3:
                    dpb["ref_feature"] = None
                dec = p_net.decompress(bit_stream, dpb, sps)
                dpb = dec["dpb"]
                recon = dpb["ref_frame"]
            recon = recon.clamp_(0, 1)
            x_hat = F.pad(recon, (-pl, -pr, -pt, -pb))
            h = recon_hash(x_hat)
            assert h == node.logs[idx]["recon_sha256"], \
                f"frame {idx}: decode hash mismatch (trajectory not deployable)"
            idx += 1
    assert len(data) * 8 == node.cum_bits, "stream accounting mismatch"
    return sha256_file(out_bin)


# ────────────────── 主流程 ──────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence_id", required=True)
    ap.add_argument("--src_path", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1920 // 1080 * 1080 or 1080)
    ap.add_argument("--q_init", type=int, required=True)
    ap.add_argument("--oracle", required=True, choices=["qonly", "resetonly", "ionly", "joint"])
    ap.add_argument("--budget_bits", type=int, required=True)
    ap.add_argument("--peak_budget_bits", type=int, required=True)
    ap.add_argument("--beam_width", type=int, default=8)
    ap.add_argument("--n_i_max", type=int, default=1)
    ap.add_argument("--n_reset_max", type=int, default=4)
    ap.add_argument("--i_spacing", type=int, default=16)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path_i", required=True)
    ap.add_argument("--model_path_p", required=True)
    ap.add_argument("--float16", action="store_true")
    ap.add_argument("--budget_floor_seg", type=int, default=0,
                    help="单段 bits 乐观下界, 默认 budget/15")
    ap.add_argument("--incumbent_schedule", default=None,
                    help="P1-R2: grid-compatible fixed baseline schedule JSON")
    ap.add_argument("--protect_incumbent", action="store_true",
                    help="P1-R2: force-keep incumbent prefix in beam every depth")
    ap.add_argument("--decision_grid", default="8:8-120",
                    help="P1-R4: e.g. '8:8-56,1:56-77,8:80-120' (event zone per-frame)")
    args = ap.parse_args()
    args.padding = (0, (args.width + 15) // 16 * 16 - args.width,
                    0, (args.height + 15) // 16 * 16 - args.height)
    from src.utils.stream_helper import get_padding_size
    args.padding = get_padding_size(args.height, args.width, 16)
    if args.budget_floor_seg == 0:
        args.budget_floor_seg = args.budget_bits // 15

    device = "cuda:0"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)
    i_net, p_net = init_models(args, device)

    ys_np, uvs_np = load_frames(args.src_path, args.width, args.height, FRAME_NUM)

    t0 = time.time()
    frontier, prune_log, inc_final = beam_search(
        args, i_net, p_net, ys_np, uvs_np, device,
        args.budget_bits, args.peak_budget_bits)
    search_s = time.time() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for j, node in enumerate(frontier):
        tag = f"{args.oracle}_{args.sequence_id}_q{args.q_init}_b{args.beam_width}_n{j}"
        out_bin = os.path.join(args.out_dir, f"{tag}.bin")
        digest = verify_trajectory(node, args, i_net, p_net, ys_np, uvs_np, device, out_bin)
        rows.append({
            "trajectory_id": tag, "oracle": args.oracle,
            "sequence_id": args.sequence_id, "q_init": args.q_init,
            "beam_width": args.beam_width,
            "actual_bits": node.cum_bits, "budget_bits": args.budget_bits,
            "within_budget": node.cum_bits <= args.budget_bits * (1 + EPS),
            "y_sse": node.cum_sse_y, "yuv_sse_weighted": node.cum_sse_yuv,
            "n_i": node.n_i, "n_reset": node.n_reset,
            "i_positions": [t for t, m, _ in node.hist if m == "I"],
            "reset_positions": [t for t, m, _ in node.hist if m == "P_RESET"],
            "q_path": [q for _, _, q in node.hist],
            "decode_verified": True, "bin_sha256": digest,
            "wall_seconds": round(search_s, 1),
        })
        with open(os.path.join(args.out_dir, f"{tag}_prune_log.json"), "w") as f:
            json.dump(prune_log, f, indent=1)

    # P1-R2: incumbent 独立行 + ENGINE FAILURE 判定
    engine_failure = False
    if inc_final is not None:
        tag = f"INCUMBENT_{args.oracle}_{args.sequence_id}_q{args.q_init}"
        out_bin = os.path.join(args.out_dir, f"{tag}.bin")
        digest = verify_trajectory(inc_final, args, i_net, p_net, ys_np, uvs_np,
                                   device, out_bin)
        rows.append({
            "trajectory_id": tag, "oracle": args.oracle + "_INCUMBENT",
            "sequence_id": args.sequence_id, "q_init": args.q_init,
            "beam_width": args.beam_width,
            "actual_bits": inc_final.cum_bits, "budget_bits": args.budget_bits,
            "within_budget": inc_final.cum_bits <= args.budget_bits * (1 + EPS),
            "y_sse": inc_final.cum_sse_y, "yuv_sse_weighted": inc_final.cum_sse_yuv,
            "n_i": inc_final.n_i, "n_reset": inc_final.n_reset,
            "i_positions": [t for t, m, _ in inc_final.hist if m == "I"],
            "reset_positions": [t for t, m, _ in inc_final.hist if m == "P_RESET"],
            "q_path": [q for _, _, q in inc_final.hist],
            "decode_verified": True, "bin_sha256": digest,
            "wall_seconds": round(search_s, 1),
        })
        best_oracle = min((r["y_sse"] for r in rows
                           if "INCUMBENT" not in r["oracle"]
                           and r["actual_bits"] <= args.budget_bits * (1 + EPS)),
                          default=None)
        if best_oracle is not None and best_oracle > inc_final.cum_sse_y:
            engine_failure = True
            print(f"ENGINE FAILURE: oracle best {best_oracle:.1f} worse than "
                  f"incumbent {inc_final.cum_sse_y:.1f} -- run marked INVALID")
        else:
            print(f"incumbent check OK: oracle best {best_oracle:.1f} "
                  f"<= incumbent {inc_final.cum_sse_y:.1f}")

    out_csv = os.path.join(args.out_dir,
                           f"frontier_{args.oracle}_{args.sequence_id}_q{args.q_init}_b{args.beam_width}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"frontier: {len(frontier)} trajectories (+1 incumbent), all decode-verified")
    within = sorted([r for r in rows if "INCUMBENT" not in r["oracle"]
                     and r["actual_bits"] <= args.budget_bits * (1 + EPS)],
                    key=lambda r: r["y_sse"])
    print("top-3 within budget (by y_sse):")
    for r in within[:3]:
        print(f"  bits={r['actual_bits']} y_sse={r['y_sse']:.1f} "
              f"resets={r['reset_positions']}")
    if engine_failure:
        print("STATUS: INVALID (engine failure)")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()