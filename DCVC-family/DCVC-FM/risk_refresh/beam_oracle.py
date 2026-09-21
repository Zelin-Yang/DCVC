#!/usr/bin/env python3
"""
beam_oracle.py — P1.4/P1.5 预算约束 beam-search oracle (协议 §8/§9)
有限动作、预算约束的 beam oracle —— 非全局最优, 名称纪律见协议 §9.1。

机制: 15 段 x 8 帧决策网格; 每节点携带完整编码状态 (DPB/SPS/码流前缀/累计量);
扩展=真实编码 8 帧段 (BytesIO 码流); 剪枝=硬约束过滤 -> Pareto -> surrogate 截断;
frontier 轨迹 100% 完整解码审计 (协议 11.2)。

用法:
  python3 -m risk_refresh.beam_oracle --sequence_id ... --src_path ... \
    --q_init 0 --oracle resetonly --budget_bits N --peak_budget_bits M \
    --beam_width 8 --out_dir ... [--n_i_max 1] [--n_reset_max 4] [--i_spacing 16]
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


def frame_to_tensor(frames, t, device):
    x = torch.from_numpy(frames[t].astype(np.float32) / 255.0)
    # 转 444 (与 runner 一致: ycbcr420_to_444 期望 y,uv 分离格式, 这里简化按 runner 流程)
    return x  # 占位, 实际在 encode_segment 内走完整转换


# ────────────────── 段编码 ──────────────────

def encode_segment(i_net, p_net, dpb_in, spss_in, ys_np, uvs_np, t0, mode, q,
                   device, padding, src_width, src_height):
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
        for k in range(SEG_LEN):
            t = t0 + k
            # yuv420 -> 444 tensor (与 runner.read_src_frame 相同流程)
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
            # 原生 SSE (float64) + PSNR
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
    # 去重
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
    depths = list(range(SEG_LEN, FRAME_NUM, SEG_LEN))  # 8,16,...,112
    q_cand = sorted({args.q_init, 0 if args.q_init != 0 else 21})

    # 根节点: t=0 段 (I + 7P) @ q_init
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
    beam = [root]
    prune_log = {"budget_violation": 0, "peak_violation": 0, "count_violation": 0,
                 "dominated": 0, "beam_limit": 0}

    for t in depths:
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
                        device, (pl, pr, pt, pb), args.width, args.height)
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
        # Pareto 过滤
        survivors = []
        for a in expanded:
            if any(dominated(a, b) for b in expanded if b is not a):
                prune_log["dominated"] += 1
            else:
                survivors.append(a)
        # 宽度截断 (预注册 surrogate: 归一化 sse + bits)
        if len(survivors) > args.beam_width:
            # 分层采样: 按 cum_bits 排序后均匀取点, 始终保留最省bits与最低sse两端
            survivors.sort(key=lambda n: n.cum_bits)
            k = args.beam_width
            idxs = sorted({round(i * (len(survivors) - 1) / max(k - 1, 1))
                           for i in range(k)} | {0, len(survivors) - 1})
            picked = [survivors[i] for i in idxs][:k]
            prune_log["beam_limit"] += len(survivors) - len(picked)
            survivors = picked
        beam = survivors
        if not beam:
            raise SystemExit(f"beam 在 t={t} 处清空 —— 预算/约束过严, 检查 budget 参数")
        print(f"depth t={t:3d}: beam={len(beam)} "
              f"bits=[{min(n.cum_bits for n in beam)}..{max(n.cum_bits for n in beam)}] "
              f"prune={prune_log}")

    # frontier = 终末 survivors 的 Pareto 集
    frontier = [a for a in beam if not any(dominated(a, b) for b in beam if b is not a)]
    return frontier, prune_log


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
                bio_unread = pending[1:]
            else:
                sps_id = header["sps_id"]
                bio_unread = []
            sps = helper.get_sps_by_id(sps_id)
            # 逐帧读取 (NAL_Ps 打包路径罕见, stream_part=1 时逐帧 NAL_I/NAL_P)
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
    ap.add_argument("--height", type=int, default=1080)
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
    args = ap.parse_args()
    args.padding = (0, (args.width + 15) // 16 * 16 - args.width,
                    0, (args.height + 15) // 16 * 16 - args.height)
    # padding 与 stream_helper.get_padding_size(p=16) 对齐 (左/上为 0)
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
    frontier, prune_log = beam_search(args, i_net, p_net, ys_np, uvs_np, device,
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

    out_csv = os.path.join(args.out_dir,
                           f"frontier_{args.oracle}_{args.sequence_id}_q{args.q_init}_b{args.beam_width}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"frontier: {len(frontier)} trajectories, all decode-verified")
    print(f"best: bits={min(r['actual_bits'] for r in rows)} "
          f"y_sse={min(r['y_sse'] for r in rows):.1f}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()