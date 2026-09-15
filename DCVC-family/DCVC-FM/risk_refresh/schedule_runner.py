#!/usr/bin/env python3
"""
schedule_runner.py — P0 显式 schedule runner (手册 Step 6)

语义约束(与 src/utils/test_helper.py 的 run_one_point_with_stream 严格一致):
  - I 帧    : SPS fa_idx=0, DPB 五字段重置
  - P_RESET : dpb["ref_feature"]=None 且 fa_idx=3 (覆盖 index_map)
  - P 帧    : fa_idx = index_map[frame_idx % rate_gop_size]
  - 解码端不读取本 schedule, 仅依据 NAL type + SPS(qp, fa_idx) 恢复状态 (E4)

输出: 逐帧审计 CSV (手册 Step 7 schema, 含 payload/sps 分解与 recon_sha256)

用法 (在 DCVC-FM 目录下, 以模块方式运行):
  python3 -m risk_refresh.schedule_runner \
    --schedule /path/to/schedule.json \
    --model_path_i checkpoints/cvpr2024_image.pth.tar \
    --model_path_p checkpoints/cvpr2024_video.pth.tar \
    --stream_path /path/to/out_dir --q 21 \
    --output_csv /path/to/per_frame.csv
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.stream_helper import (
    get_padding_size, get_state_dict, SPSHelper, NalType,
    write_sps, read_header, read_sps_remaining, read_ip_remaining,
)
from src.transforms.functional import ycbcr444_to_420, ycbcr420_to_444
from src.utils.metrics import calc_psnr, calc_msssim
from src.utils.video_reader import YUVReader

from risk_refresh.schedule_io import validate

INDEX_MAP = [0, 1, 0, 2, 0, 2, 0, 2]  # 与 test_helper.py 完全一致

CSV_FIELDS = [
    "schedule_id", "sequence_id", "frame_idx", "poc", "mode", "q_index", "fa_idx",
    "payload_bits", "sps_bits", "actual_total_bits",
    "psnr", "psnr_y", "psnr_u", "psnr_v", "ms_ssim",
    "enc_ms", "dec_ms", "ref_age", "reset_age",
    "recon_sha256",
    "codec_commit", "checkpoint_hash",
    "budget_remaining", "risk_pred", "action_value", "oracle_label",  # Phase-2 预留
]


def recon_hash(x_hat):
    arr = x_hat.squeeze(0).cpu().numpy().astype(np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def init_models(args, device):
    i_state = get_state_dict(args.model_path_i)
    i_net = DMCI(ec_thread=False, stream_part=1, inplace=True)
    i_net.load_state_dict(i_state)
    i_net = i_net.to(device).eval()

    p_state = get_state_dict(args.model_path_p)
    p_net = DMC(ec_thread=False, stream_part=1, inplace=True)
    p_net.load_state_dict(p_state)
    p_net = p_net.to(device).eval()

    i_net.update(force=True)
    p_net.update(force=True)
    if args.float16:
        i_net.half()
        p_net.half()
    return i_net, p_net


def read_src_frame(src_reader, device, float16):
    y, uv = src_reader.read_one_frame(dst_format="420")
    yuv = ycbcr420_to_444(y, uv)
    x = torch.from_numpy(yuv).type(torch.FloatTensor).unsqueeze(0)
    if float16:
        x = x.to(torch.float16)
    return x.to(device), y[0, :, :], uv[0, :, :], uv[1, :, :]


def calc_distortion(x_hat, y, u, v, calc_ssim_flag):
    yuv_rec = x_hat.squeeze(0).cpu().numpy()
    y_rec, uv_rec = ycbcr444_to_420(yuv_rec)
    psnr_y = calc_psnr(y, y_rec[0, :, :], data_range=1)
    psnr_u = calc_psnr(u, uv_rec[0, :, :], data_range=1)
    psnr_v = calc_psnr(v, uv_rec[1, :, :], data_range=1)
    psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
    if calc_ssim_flag:
        ssim = (6 * calc_msssim(y, y_rec[0, :, :], data_range=1)
                + calc_msssim(u, uv_rec[0, :, :], data_range=1)
                + calc_msssim(v, uv_rec[1, :, :], data_range=1)) / 8
    else:
        ssim = 0.0
    return psnr, psnr_y, psnr_u, psnr_v, ssim


def run_schedule(args):
    with open(args.schedule, encoding="utf-8") as f:
        schedule = json.load(f)
    actions = validate(schedule)          # 任何结构问题在此拒绝

    seq = schedule["sequence"]
    frame_num = seq["frame_num"]
    pic_h, pic_w = seq["height"], seq["width"]
    schedule_id = schedule["schedule_id"]

    # codec_commit 一致性检查 (schedule 若声明了 commit 则必须匹配)
    fm_dir = os.path.dirname(os.path.abspath(__file__)) + "/../.."
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=fm_dir, text=True).strip()
    if schedule.get("codec_commit") and schedule["codec_commit"] not in ("", actual_commit):
        raise SystemExit(
            f"codec_commit mismatch: schedule={schedule['codec_commit']} actual={actual_commit}")

    device = "cuda:0" if args.cuda else "cpu"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(0)

    i_net, p_net = init_models(args, device)

    src_reader = YUVReader(seq["src_path"], pic_w, pic_h)
    padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_h, pic_w, 16)

    os.makedirs(args.stream_path, exist_ok=True)
    bin_path = Path(args.stream_path) / f"{seq['sequence_id']}_q{args.q}.bin"
    output_file = bin_path.open("wb")
    sps_helper = SPSHelper()
    outstanding_sps_bytes = 0

    rows = []
    psnr_enc_log = []          # 编码侧 PSNR, 解码后逐一断言一致 (E2)
    dpb = None
    last_i = 0
    last_refresh = 0

    with torch.no_grad():
        # ─── 编码循环 ───
        for t in range(frame_num):
            act = actions[t]
            mode, q = act["mode"], act["q_index"]
            frame_start = time.time()
            x, y, u, v = read_src_frame(src_reader, device, args.float16)
            x_padded = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            if mode == "I":
                sps = {"sps_id": -1, "height": pic_h, "width": pic_w, "qp": q, "fa_idx": 0}
                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps["sps_id"] = sps_id
                if sps_new:
                    outstanding_sps_bytes += write_sps(output_file, sps)
                    if args.verbose >= 2:
                        print("new sps", sps)
                result = i_net.encode(x_padded, q, sps_id, output_file)
                dpb = {"ref_frame": result["x_hat"], "ref_feature": None,
                       "ref_mv_feature": None, "ref_y": None, "ref_mv_y": None}
                recon = result["x_hat"]
                payload_bits = result["bit"]
                fa_idx = 0
                last_i = t
                last_refresh = t
            else:
                fa_idx = INDEX_MAP[t % args.rate_gop_size]
                if mode == "P_RESET":
                    dpb["ref_feature"] = None
                    fa_idx = 3
                    last_refresh = t
                sps = {"sps_id": -1, "height": pic_h, "width": pic_w, "qp": q, "fa_idx": fa_idx}
                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps["sps_id"] = sps_id
                if sps_new:
                    outstanding_sps_bytes += write_sps(output_file, sps)
                    if args.verbose >= 2:
                        print("new sps", sps)
                result = p_net.encode(x_padded, dpb, q, fa_idx, sps_id, output_file)
                dpb = result["dpb"]
                recon = dpb["ref_frame"]
                payload_bits = result["bit"]

            recon = recon.clamp_(0, 1)
            x_hat = F.pad(recon, (-padding_l, -padding_r, -padding_t, -padding_b))
            enc_ms = (time.time() - frame_start) * 1000
            psnr, psnr_y, psnr_u, psnr_v, msssim = calc_distortion(
                x_hat, y, u, v, args.calc_ssim)
            psnr_enc_log.append(psnr)

            rows.append({
                "schedule_id": schedule_id, "sequence_id": seq["sequence_id"],
                "frame_idx": t, "poc": t, "mode": mode, "q_index": q, "fa_idx": fa_idx,
                "payload_bits": payload_bits,
                "sps_bits": outstanding_sps_bytes * 8,
                "actual_total_bits": payload_bits + outstanding_sps_bytes * 8,
                "psnr": psnr, "psnr_y": psnr_y, "psnr_u": psnr_u, "psnr_v": psnr_v,
                "ms_ssim": msssim,
                "enc_ms": round(enc_ms, 1), "dec_ms": "",
                "ref_age": t - last_i, "reset_age": t - last_refresh,
                "recon_sha256": recon_hash(x_hat),
                "codec_commit": actual_commit,
                "checkpoint_hash": args.checkpoint_hash,
                "budget_remaining": "", "risk_pred": "", "action_value": "",
                "oracle_label": "",
            })
            outstanding_sps_bytes = 0

            if args.verbose >= 2:
                print(f"frame {t} encoded, {enc_ms/1000:.3f} s, bits: {rows[-1]['actual_total_bits']}, "
                      f"PSNR: {psnr:.4f}, MS-SSIM: {msssim:.4f}")

        src_reader.close()
        output_file.close()

        # ─── 解码循环(不读取 schedule, 仅 NAL/SPS) ───
        sps_helper = SPSHelper()
        input_file = bin_path.open("rb")
        src_reader = YUVReader(seq["src_path"], pic_w, pic_h)
        decoded_n = 0
        pending_spss = []
        p_dec_time = 0.0
        p_dec_count = 0
        while decoded_n < frame_num:
            new_stream = False
            if len(pending_spss) == 0:
                header = read_header(input_file)
                if header["nal_type"] == NalType.NAL_SPS:
                    sps = read_sps_remaining(input_file, header["sps_id"])
                    sps_helper.add_sps_by_id(sps)
                    if args.verbose >= 2:
                        print("new sps", sps)
                    continue
                if header["nal_type"] == NalType.NAL_Ps:
                    pending_spss = header["sps_ids"][1:]
                    sps_id = header["sps_ids"][0]
                else:
                    sps_id = header["sps_id"]
                new_stream = True
            else:
                sps_id = pending_spss[0]
                pending_spss.pop(0)

            sps = sps_helper.get_sps_by_id(sps_id)
            if new_stream:
                bit_stream = read_ip_remaining(input_file)
            else:
                bit_stream = None

            frame_start = time.time()
            x, y, u, v = read_src_frame(src_reader, device, args.float16)

            if header["nal_type"] == NalType.NAL_I:
                decoded = i_net.decompress(bit_stream, sps)
                dpb = {"ref_frame": decoded["x_hat"], "ref_feature": None,
                       "ref_mv_feature": None, "ref_y": None, "ref_mv_y": None}
                recon = decoded["x_hat"]
            else:
                if sps["fa_idx"] == 3:
                    dpb["ref_feature"] = None
                decoded = p_net.decompress(bit_stream, dpb, sps)
                dpb = decoded["dpb"]
                recon = dpb["ref_frame"]
                p_dec_time += decoded["decoding_time"]
                p_dec_count += 1

            recon = recon.clamp_(0, 1)
            x_hat = F.pad(recon, (-padding_l, -padding_r, -padding_t, -padding_b))
            dec_ms = (time.time() - frame_start) * 1000
            psnr, psnr_y, psnr_u, psnr_v, msssim = calc_distortion(
                x_hat, y, u, v, args.calc_ssim)

            # E2 强化校验: 编解码逐帧 PSNR + recon hash 双重一致
            assert abs(psnr - psnr_enc_log[decoded_n]) < 1e-9, \
                f"frame {decoded_n}: enc PSNR {psnr_enc_log[decoded_n]} != dec PSNR {psnr}"
            assert recon_hash(x_hat) == rows[decoded_n]["recon_sha256"], \
                f"frame {decoded_n}: recon hash mismatch (encoder/decoder state desync)"

            rows[decoded_n]["dec_ms"] = round(dec_ms, 1)
            rows[decoded_n]["psnr"] = psnr  # 以解码侧为准回填
            decoded_n += 1

            if args.verbose >= 2:
                print(f"frame {decoded_n - 1} decoded, {dec_ms/1000:.3f} s, "
                      f"bits: {0 if bit_stream is None else len(bit_stream) * 8}, "
                      f"PSNR: {psnr:.4f}")

        input_file.close()
        src_reader.close()

    # ─── E5 记账校验 ───
    file_bits = bin_path.stat().st_size * 8
    total_bits = sum(r["actual_total_bits"] for r in rows)
    assert total_bits == file_bits, f"E5 accounting mismatch: frames={total_bits} file={file_bits}"

    # ─── 写 CSV ───
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if f.tell() == 0:
            w.writeheader()
        w.writerows(rows)

    print(f"schedule {schedule_id}: {frame_num} frames, "
          f"avg P dec {p_dec_time / max(p_dec_count, 1) * 1000:.0f} ms")
    print(f"E2 PASS (PSNR + recon hash); E5 PASS ({total_bits} bits == 8 x {bin_path.name})")
    print(f"rows appended -> {args.output_csv}")


def parse_args():
    ap = argparse.ArgumentParser(description="P0 schedule runner (risk_refresh)")
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--model_path_i", required=True)
    ap.add_argument("--model_path_p", required=True)
    ap.add_argument("--stream_path", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--q", type=int, required=True, help="q_index tag for bin naming")
    ap.add_argument("--rate_gop_size", type=int, default=8, choices=[4, 8])
    ap.add_argument("--calc_ssim", action="store_true")
    ap.add_argument("--float16", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--verbose", type=int, default=2)
    ap.add_argument("--checkpoint_hash", default="", help="e.g. sha256 of video ckpt for log")
    return ap.parse_args()


if __name__ == "__main__":
    run_schedule(parse_args())