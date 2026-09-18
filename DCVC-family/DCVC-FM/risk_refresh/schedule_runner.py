# #!/usr/bin/env python3
# """
# schedule_runner.py — P0 显式 schedule runner (手册 Step 6)

# 语义约束(与 src/utils/test_helper.py 的 run_one_point_with_stream 严格一致):
#   - I 帧    : SPS fa_idx=0, DPB 五字段重置
#   - P_RESET : dpb["ref_feature"]=None 且 fa_idx=3 (覆盖 index_map)
#   - P 帧    : fa_idx = index_map[frame_idx % rate_gop_size]
#   - 解码端不读取本 schedule, 仅依据 NAL type + SPS(qp, fa_idx) 恢复状态 (E4)

# 输出: 逐帧审计 CSV (手册 Step 7 schema, 含 payload/sps 分解与 recon_sha256)

# 用法 (在 DCVC-FM 目录下, 以模块方式运行):
#   python3 -m risk_refresh.schedule_runner \
#     --schedule /path/to/schedule.json \
#     --model_path_i checkpoints/cvpr2024_image.pth.tar \
#     --model_path_p checkpoints/cvpr2024_video.pth.tar \
#     --stream_path /path/to/out_dir --q 21 \
#     --output_csv /path/to/per_frame.csv
# """

# import argparse
# import csv
# import hashlib
# import json
# import os
# import subprocess
# import time
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn.functional as F

# from src.models.video_model import DMC
# from src.models.image_model import DMCI
# from src.utils.stream_helper import (
#     get_padding_size, get_state_dict, SPSHelper, NalType,
#     write_sps, read_header, read_sps_remaining, read_ip_remaining,
# )
# from src.transforms.functional import ycbcr444_to_420, ycbcr420_to_444
# from src.utils.metrics import calc_psnr, calc_msssim
# from src.utils.video_reader import YUVReader

# from risk_refresh.schedule_io import validate

# INDEX_MAP = [0, 1, 0, 2, 0, 2, 0, 2]  # 与 test_helper.py 完全一致

# CSV_FIELDS = [
#     "schedule_id", "sequence_id", "frame_idx", "poc", "mode", "q_index", "fa_idx",
#     "payload_bits", "sps_bits", "actual_total_bits",
#     "psnr", "psnr_y", "psnr_u", "psnr_v", "ms_ssim",
#     "enc_ms", "dec_ms", "ref_age", "reset_age",
#     "recon_sha256",
#     "codec_commit", "checkpoint_hash",
#     "budget_remaining", "risk_pred", "action_value", "oracle_label",  # Phase-2 预留
# ]


# def recon_hash(x_hat):
#     arr = x_hat.squeeze(0).cpu().numpy().astype(np.float32)
#     return hashlib.sha256(arr.tobytes()).hexdigest()


# def init_models(args, device):
#     i_state = get_state_dict(args.model_path_i)
#     i_net = DMCI(ec_thread=False, stream_part=1, inplace=True)
#     i_net.load_state_dict(i_state)
#     i_net = i_net.to(device).eval()

#     p_state = get_state_dict(args.model_path_p)
#     p_net = DMC(ec_thread=False, stream_part=1, inplace=True)
#     p_net.load_state_dict(p_state)
#     p_net = p_net.to(device).eval()

#     i_net.update(force=True)
#     p_net.update(force=True)
#     if args.float16:
#         i_net.half()
#         p_net.half()
#     return i_net, p_net


# def read_src_frame(src_reader, device, float16):
#     y, uv = src_reader.read_one_frame(dst_format="420")
#     yuv = ycbcr420_to_444(y, uv)
#     x = torch.from_numpy(yuv).type(torch.FloatTensor).unsqueeze(0)
#     if float16:
#         x = x.to(torch.float16)
#     return x.to(device), y[0, :, :], uv[0, :, :], uv[1, :, :]


# def calc_distortion(x_hat, y, u, v, calc_ssim_flag):
#     yuv_rec = x_hat.squeeze(0).cpu().numpy()
#     y_rec, uv_rec = ycbcr444_to_420(yuv_rec)
#     psnr_y = calc_psnr(y, y_rec[0, :, :], data_range=1)
#     psnr_u = calc_psnr(u, uv_rec[0, :, :], data_range=1)
#     psnr_v = calc_psnr(v, uv_rec[1, :, :], data_range=1)
#     psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
#     if calc_ssim_flag:
#         ssim = (6 * calc_msssim(y, y_rec[0, :, :], data_range=1)
#                 + calc_msssim(u, uv_rec[0, :, :], data_range=1)
#                 + calc_msssim(v, uv_rec[1, :, :], data_range=1)) / 8
#     else:
#         ssim = 0.0
#     return psnr, psnr_y, psnr_u, psnr_v, ssim


# def run_schedule(args):
#     with open(args.schedule, encoding="utf-8") as f:
#         schedule = json.load(f)
#     actions = validate(schedule)          # 任何结构问题在此拒绝

#     seq = schedule["sequence"]
#     frame_num = seq["frame_num"]
#     pic_h, pic_w = seq["height"], seq["width"]
#     schedule_id = schedule["schedule_id"]

#     # codec_commit 一致性检查 (schedule 若声明了 commit 则必须匹配)
#     fm_dir = os.path.dirname(os.path.abspath(__file__)) + "/../.."
#     actual_commit = subprocess.check_output(
#         ["git", "rev-parse", "HEAD"], cwd=fm_dir, text=True).strip()
#     if schedule.get("codec_commit") and schedule["codec_commit"] not in ("", actual_commit):
#         raise SystemExit(
#             f"codec_commit mismatch: schedule={schedule['codec_commit']} actual={actual_commit}")

#     device = "cuda:0" if args.cuda else "cpu"
#     torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True)
#     torch.manual_seed(0)
#     torch.set_num_threads(1)
#     np.random.seed(0)

#     i_net, p_net = init_models(args, device)

#     src_reader = YUVReader(seq["src_path"], pic_w, pic_h)
#     padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_h, pic_w, 16)

#     os.makedirs(args.stream_path, exist_ok=True)
#     bin_path = Path(args.stream_path) / f"{seq['sequence_id']}_q{args.q}.bin"
#     output_file = bin_path.open("wb")
#     sps_helper = SPSHelper()
#     outstanding_sps_bytes = 0

#     rows = []
#     psnr_enc_log = []          # 编码侧 PSNR, 解码后逐一断言一致 (E2)
#     dpb = None
#     last_i = 0
#     last_refresh = 0

#     with torch.no_grad():
#         # ─── 编码循环 ───
#         for t in range(frame_num):
#             act = actions[t]
#             mode, q = act["mode"], act["q_index"]
#             frame_start = time.time()
#             x, y, u, v = read_src_frame(src_reader, device, args.float16)
#             x_padded = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

#             if mode == "I":
#                 sps = {"sps_id": -1, "height": pic_h, "width": pic_w, "qp": q, "fa_idx": 0}
#                 sps_id, sps_new = sps_helper.get_sps_id(sps)
#                 sps["sps_id"] = sps_id
#                 if sps_new:
#                     outstanding_sps_bytes += write_sps(output_file, sps)
#                     if args.verbose >= 2:
#                         print("new sps", sps)
#                 result = i_net.encode(x_padded, q, sps_id, output_file)
#                 dpb = {"ref_frame": result["x_hat"], "ref_feature": None,
#                        "ref_mv_feature": None, "ref_y": None, "ref_mv_y": None}
#                 recon = result["x_hat"]
#                 payload_bits = result["bit"]
#                 fa_idx = 0
#                 last_i = t
#                 last_refresh = t
#             else:
#                 fa_idx = INDEX_MAP[t % args.rate_gop_size]
#                 if mode == "P_RESET":
#                     dpb["ref_feature"] = None
#                     fa_idx = 3
#                     last_refresh = t
#                 sps = {"sps_id": -1, "height": pic_h, "width": pic_w, "qp": q, "fa_idx": fa_idx}
#                 sps_id, sps_new = sps_helper.get_sps_id(sps)
#                 sps["sps_id"] = sps_id
#                 if sps_new:
#                     outstanding_sps_bytes += write_sps(output_file, sps)
#                     if args.verbose >= 2:
#                         print("new sps", sps)
#                 result = p_net.encode(x_padded, dpb, q, fa_idx, sps_id, output_file)
#                 dpb = result["dpb"]
#                 recon = dpb["ref_frame"]
#                 payload_bits = result["bit"]

#             recon = recon.clamp_(0, 1)
#             x_hat = F.pad(recon, (-padding_l, -padding_r, -padding_t, -padding_b))
#             enc_ms = (time.time() - frame_start) * 1000
#             psnr, psnr_y, psnr_u, psnr_v, msssim = calc_distortion(
#                 x_hat, y, u, v, args.calc_ssim)
#             psnr_enc_log.append(psnr)

#             rows.append({
#                 "schedule_id": schedule_id, "sequence_id": seq["sequence_id"],
#                 "frame_idx": t, "poc": t, "mode": mode, "q_index": q, "fa_idx": fa_idx,
#                 "payload_bits": payload_bits,
#                 "sps_bits": outstanding_sps_bytes * 8,
#                 "actual_total_bits": payload_bits + outstanding_sps_bytes * 8,
#                 "psnr": psnr, "psnr_y": psnr_y, "psnr_u": psnr_u, "psnr_v": psnr_v,
#                 "ms_ssim": msssim,
#                 "enc_ms": round(enc_ms, 1), "dec_ms": "",
#                 "ref_age": t - last_i, "reset_age": t - last_refresh,
#                 "recon_sha256": recon_hash(x_hat),
#                 "codec_commit": actual_commit,
#                 "checkpoint_hash": args.checkpoint_hash,
#                 "budget_remaining": "", "risk_pred": "", "action_value": "",
#                 "oracle_label": "",
#             })
#             outstanding_sps_bytes = 0

#             if args.verbose >= 2:
#                 print(f"frame {t} encoded, {enc_ms/1000:.3f} s, bits: {rows[-1]['actual_total_bits']}, "
#                       f"PSNR: {psnr:.4f}, MS-SSIM: {msssim:.4f}")

#         src_reader.close()
#         output_file.close()

#         # ─── 解码循环(不读取 schedule, 仅 NAL/SPS) ───
#         sps_helper = SPSHelper()
#         input_file = bin_path.open("rb")
#         src_reader = YUVReader(seq["src_path"], pic_w, pic_h)
#         decoded_n = 0
#         pending_spss = []
#         p_dec_time = 0.0
#         p_dec_count = 0
#         while decoded_n < frame_num:
#             new_stream = False
#             if len(pending_spss) == 0:
#                 header = read_header(input_file)
#                 if header["nal_type"] == NalType.NAL_SPS:
#                     sps = read_sps_remaining(input_file, header["sps_id"])
#                     sps_helper.add_sps_by_id(sps)
#                     if args.verbose >= 2:
#                         print("new sps", sps)
#                     continue
#                 if header["nal_type"] == NalType.NAL_Ps:
#                     pending_spss = header["sps_ids"][1:]
#                     sps_id = header["sps_ids"][0]
#                 else:
#                     sps_id = header["sps_id"]
#                 new_stream = True
#             else:
#                 sps_id = pending_spss[0]
#                 pending_spss.pop(0)

#             sps = sps_helper.get_sps_by_id(sps_id)
#             if new_stream:
#                 bit_stream = read_ip_remaining(input_file)
#             else:
#                 bit_stream = None

#             frame_start = time.time()
#             x, y, u, v = read_src_frame(src_reader, device, args.float16)

#             if header["nal_type"] == NalType.NAL_I:
#                 decoded = i_net.decompress(bit_stream, sps)
#                 dpb = {"ref_frame": decoded["x_hat"], "ref_feature": None,
#                        "ref_mv_feature": None, "ref_y": None, "ref_mv_y": None}
#                 recon = decoded["x_hat"]
#             else:
#                 if sps["fa_idx"] == 3:
#                     dpb["ref_feature"] = None
#                 decoded = p_net.decompress(bit_stream, dpb, sps)
#                 dpb = decoded["dpb"]
#                 recon = dpb["ref_frame"]
#                 p_dec_time += decoded["decoding_time"]
#                 p_dec_count += 1

#             recon = recon.clamp_(0, 1)
#             x_hat = F.pad(recon, (-padding_l, -padding_r, -padding_t, -padding_b))
#             dec_ms = (time.time() - frame_start) * 1000
#             psnr, psnr_y, psnr_u, psnr_v, msssim = calc_distortion(
#                 x_hat, y, u, v, args.calc_ssim)

#             # E2 强化校验: 编解码逐帧 PSNR + recon hash 双重一致
#             assert abs(psnr - psnr_enc_log[decoded_n]) < 1e-9, \
#                 f"frame {decoded_n}: enc PSNR {psnr_enc_log[decoded_n]} != dec PSNR {psnr}"
#             assert recon_hash(x_hat) == rows[decoded_n]["recon_sha256"], \
#                 f"frame {decoded_n}: recon hash mismatch (encoder/decoder state desync)"

#             rows[decoded_n]["dec_ms"] = round(dec_ms, 1)
#             rows[decoded_n]["psnr"] = psnr  # 以解码侧为准回填
#             decoded_n += 1

#             if args.verbose >= 2:
#                 print(f"frame {decoded_n - 1} decoded, {dec_ms/1000:.3f} s, "
#                       f"bits: {0 if bit_stream is None else len(bit_stream) * 8}, "
#                       f"PSNR: {psnr:.4f}")

#         input_file.close()
#         src_reader.close()

#     # ─── E5 记账校验 ───
#     file_bits = bin_path.stat().st_size * 8
#     total_bits = sum(r["actual_total_bits"] for r in rows)
#     assert total_bits == file_bits, f"E5 accounting mismatch: frames={total_bits} file={file_bits}"

#     # ─── 写 CSV ───
#     os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
#     with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
#         if f.tell() == 0:
#             w.writeheader()
#         w.writerows(rows)

#     print(f"schedule {schedule_id}: {frame_num} frames, "
#           f"avg P dec {p_dec_time / max(p_dec_count, 1) * 1000:.0f} ms")
#     print(f"E2 PASS (PSNR + recon hash); E5 PASS ({total_bits} bits == 8 x {bin_path.name})")
#     print(f"rows appended -> {args.output_csv}")


# def parse_args():
#     ap = argparse.ArgumentParser(description="P0 schedule runner (risk_refresh)")
#     ap.add_argument("--schedule", required=True)
#     ap.add_argument("--model_path_i", required=True)
#     ap.add_argument("--model_path_p", required=True)
#     ap.add_argument("--stream_path", required=True)
#     ap.add_argument("--output_csv", required=True)
#     ap.add_argument("--q", type=int, required=True, help="q_index tag for bin naming")
#     ap.add_argument("--rate_gop_size", type=int, default=8, choices=[4, 8])
#     ap.add_argument("--calc_ssim", action="store_true")
#     ap.add_argument("--float16", action="store_true")
#     ap.add_argument("--cuda", action="store_true")
#     ap.add_argument("--verbose", type=int, default=2)
#     ap.add_argument("--checkpoint_hash", default="", help="e.g. sha256 of video ckpt for log")
#     return ap.parse_args()


# if __name__ == "__main__":
#     run_schedule(parse_args())

# ====================================================================
#!/usr/bin/env python3
"""
schedule_runner.py — P0 显式 schedule runner + P1 DPB 快照分支 (手册 Step 6, 协议 §5)

P0 语义(与 test_helper.run_one_point_with_stream 严格一致, E3 已字节级验证):
  I: SPS fa_idx=0 + DPB 五字段重置;  P_RESET: ref_feature=None + fa_idx=3;
  P: fa_idx = INDEX_MAP[t % gop];  解码端仅 NAL/SPS (E4);  E2 双断言;  E5 记账断言。

P1 快照(协议 §5):
  --snapshot_dir + --snapshot_frames "t1 t2 ...": 在执行帧 t 的动作**之前**保存 pre-frame_t
  --resume_snapshot + --branch_schedule: 从快照恢复, prefix 复制续写, 执行 branch 后缀
  快照内容: encoder DPB (decoder 按 E2 与之恒等, 恢复时同一对象重建)、SPSHelper 全表、
           prefix.bin、累计 bits、轨迹状态、源/checkpoint/commit/protocol 哈希、schema_version

用法 (DCVC-FM 目录下):
  # P0 常规运行
  python3 -m risk_refresh.schedule_runner --schedule s.json --model_path_i ... --model_path_p ... \
      --stream_path out --q 21 --output_csv log.csv --cuda --calc_ssim
  # P1.1 快照 + 等价自验: 先带快照跑完整基线, 再从各快照恢复跑同一后缀
  python3 -m risk_refresh.schedule_runner ... --snapshot_dir snaps/ --snapshot_frames "1 32 62 80"
  python3 -m risk_refresh.schedule_runner ... --resume_snapshot snaps/pre_frame_0032 \
      --branch_schedule B0_suffix_from32.json --branch_tag eq32 --stream_path out --output_csv log.csv
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
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

from risk_refresh.schedule_io import validate, MODES

INDEX_MAP = [0, 1, 0, 2, 0, 2, 0, 2]
SNAPSHOT_SCHEMA_VERSION = 1

CSV_FIELDS = [
    "schedule_id", "sequence_id", "frame_idx", "poc", "mode", "q_index", "fa_idx",
    "payload_bits", "sps_bits", "actual_total_bits",
    "psnr", "psnr_y", "psnr_u", "psnr_v", "ms_ssim",
    "enc_ms", "dec_ms", "ref_age", "reset_age",
    "recon_sha256", "codec_commit", "checkpoint_hash",
    "budget_remaining", "risk_pred", "action_value", "oracle_label",
    "prefix_actual_bits",   # P1: resume 分支时记录共同前缀 bits
    "run_role",             # P1: "full" | "branch"
]


# ───────────────────────────── 基础工具 ─────────────────────────────

def sha256_file(path, chunk=16 * 1024 * 1024):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            d.update(b)
    return d.hexdigest()


def recon_hash(x_hat):
    arr = x_hat.squeeze(0).cpu().numpy().astype(np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def tensor_pack(dpb):
    """DPB -> CPU 纯字典; None 保留为 None; 记录每字段 shape/dtype/hash。"""
    out, meta = {}, {}
    for k, v in dpb.items():
        if v is None:
            out[k] = None
            meta[k] = {"is_none": True}
        else:
            t = v.detach().cpu().clone()
            out[k] = t
            meta[k] = {
                "is_none": False,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "sha256": hashlib.sha256(
                    t.contiguous().numpy().astype(np.float32).tobytes()).hexdigest(),
            }
    return out, meta


def tensor_unpack(pack, meta, device):
    dpb = {}
    for k, v in pack.items():
        if meta[k]["is_none"]:
            dpb[k] = None
        else:
            t = v.to(device)
            assert list(t.shape) == meta[k]["shape"], f"snapshot {k} shape mismatch"
            assert str(t.dtype) == meta[k]["dtype"], f"snapshot {k} dtype mismatch"
            dpb[k] = t
    return dpb


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


# ───────────────────────────── 快照 ─────────────────────────────

def save_snapshot(dir_path, t, dpb, sps_helper, prefix_src_path, cum_bits,
                  trajectory, schedule_id, seq, codec_commit, checkpoint_hash,
                  device_note):
    """保存 pre-frame t 状态(执行帧 t 动作前)。原子写: 临时目录后 rename。"""
    tmp = Path(str(dir_path) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    pack, tmeta = tensor_pack(dpb)
    torch.save(pack, tmp / "encoder_dpb.pt")

    with open(tmp / "sps_state.json", "w") as f:
        json.dump({"spss": sps_helper.spss}, f)

    prefix_bytes = Path(prefix_src_path).stat().st_size
    shutil.copyfile(prefix_src_path, tmp / "prefix.bin")

    meta = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "semantics": "pre-frame: encoding of frames [0, t) completed; next action frame is t",
        "next_frame_idx": t,
        "schedule_id": schedule_id,
        "sequence": seq,
        "cum_actual_bits": cum_bits,
        "prefix_bytes": prefix_bytes,
        "trajectory": trajectory,
        "tensor_meta": tmeta,
        "codec_commit": codec_commit,
        "checkpoint_hash": checkpoint_hash,
        "device_note": device_note,
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prefix_sha256": sha256_file(prefix_src_path),
    }
    with open(tmp / "metadata.json", "w") as f:
        json.dump(meta, f, indent=1)

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.replace(tmp, dir_path)
    return meta


def load_snapshot(snap_dir, device, expect_schedule_id=None, expect_seq_id=None):
    snap_dir = Path(snap_dir)
    with open(snap_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["schema_version"] == SNAPSHOT_SCHEMA_VERSION, "snapshot schema mismatch"

    if expect_schedule_id and meta["schedule_id"] != expect_schedule_id:
        raise SystemExit(f"snapshot schedule mismatch: {meta['schedule_id']}")
    if expect_seq_id and meta["sequence"]["sequence_id"] != expect_seq_id:
        raise SystemExit(f"snapshot sequence mismatch: {meta['sequence']['sequence_id']}")

    pack = torch.load(snap_dir / "encoder_dpb.pt", map_location="cpu")
    dpb = tensor_unpack(pack, meta["tensor_meta"], device)

    with open(snap_dir / "sps_state.json") as f:
        sps_state = json.load(f)
    helper = SPSHelper()
    helper.spss = [dict(s) for s in sps_state["spss"]]

    return meta, dpb, helper


def validate_branch_schedule(branch, start_idx, frame_num):
    """branch schedule 覆盖 [start_idx, frame_num), 逐帧唯一, mode/q 合法。"""
    table = {}
    for a in branch["actions"]:
        t = a["frame_idx"]
        assert start_idx <= t < frame_num, f"branch frame {t} outside [{start_idx},{frame_num})"
        assert t not in table, f"duplicate branch action at {t}"
        assert a["mode"] in MODES, f"bad mode {a['mode']}"
        q = int(a["q_index"])
        assert 0 <= q <= 63
        table[t] = {"mode": a["mode"], "q_index": q, "reason": a["reason"]}
    missing = [t for t in range(start_idx, frame_num) if t not in table]
    assert not missing, f"branch missing frames {missing[:10]}..."
    return table


# ───────────────────────────── 主流程 ─────────────────────────────

def run_schedule(args):
    codec_dir = os.path.dirname(os.path.abspath(__file__)) + "/../.."
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=codec_dir, text=True).strip()

    device = "cuda:0" if args.cuda else "cpu"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(0)

    i_net, p_net = init_models(args, device)

    resume_meta = None
    if args.resume_snapshot:
        # ── P1 分支模式 ──
        with open(args.branch_schedule, encoding="utf-8") as f:
            branch = json.load(f)
        seq = branch["sequence"]
        frame_num = seq["frame_num"]
        pic_h, pic_w = seq["height"], seq["width"]
        schedule_id = branch.get("schedule_id", "branch")

        resume_meta, dpb, sps_helper = load_snapshot(
            args.resume_snapshot, device,
            expect_seq_id=seq["sequence_id"])
        start_idx = resume_meta["next_frame_idx"]
        actions = validate_branch_schedule(branch, start_idx, frame_num)
        prefix_bits = resume_meta["cum_actual_bits"]
        trajectory = dict(resume_meta["trajectory"])
        run_role = "branch"
        tag = args.branch_tag or f"from{start_idx:04d}"
    else:
        # ── P0/P1 完整模式 ──
        with open(args.schedule, encoding="utf-8") as f:
            schedule = json.load(f)
        actions_full = validate(schedule)
        seq = schedule["sequence"]
        frame_num = seq["frame_num"]
        pic_h, pic_w = seq["height"], seq["width"]
        schedule_id = schedule["schedule_id"]
        if schedule.get("codec_commit") and schedule["codec_commit"] not in ("", actual_commit):
            raise SystemExit("codec_commit mismatch")
        start_idx = 0
        actions = actions_full
        dpb = None
        sps_helper = SPSHelper()
        prefix_bits = 0
        trajectory = {"last_i": 0, "last_refresh": 0, "i_count": 0, "reset_count": 0,
                      "last_q": None, "last_fa": None}
        run_role = "full"
        tag = ""

    src_reader = YUVReader(seq["src_path"], pic_w, pic_h)
    if resume_meta and start_idx > 0:
        # P1.1-fix: 对齐源读取器 —— 丢弃前缀帧, 使 read_one_frame 与 frame_idx 一致
        for _ in range(start_idx):
            src_reader.read_one_frame(dst_format="420")
    padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_h, pic_w, 16)

    os.makedirs(args.stream_path, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    bin_path = Path(args.stream_path) / f"{seq['sequence_id']}_q{args.q}{suffix}.bin"

    if resume_meta:
        # 前缀字节复制续写 (协议 5.3-7)
        shutil.copyfile(Path(args.resume_snapshot) / "prefix.bin", bin_path)
        output_file = bin_path.open("ab")
        cum_bits = prefix_bits
    else:
        output_file = bin_path.open("wb")
        cum_bits = 0

    outstanding_sps_bytes = 0
    rows = []
    psnr_enc_log = {}
    snap_frames = set(args.snapshot_frames) if args.snapshot_frames else set()

    with torch.no_grad():
        # ─── 编码循环 ───
        for t in range(start_idx, frame_num):
            act = actions[t]
            mode, q = act["mode"], act["q_index"]

            # P1.1: pre-frame t 快照(执行动作前)
            if t in snap_frames and args.snapshot_dir:
                snap_dir = Path(args.snapshot_dir) / f"pre_frame_{t:04d}"
                assert dpb is not None, f"cannot snapshot at frame {t}: dpb is None"
                output_file.flush()  # P1.1-fix: 缓冲区落盘后再复制前缀, 否则 prefix.bin 截断导致合并码流损坏
                save_snapshot(snap_dir, t, dpb, sps_helper, bin_path, cum_bits,
                              trajectory, schedule_id, seq, actual_commit,
                              args.checkpoint_hash,
                              device_note=f"device={device},float16={args.float16}")
                if args.verbose >= 1:
                    print(f"snapshot saved: pre_frame_{t:04d}")

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
                trajectory.update(last_i=t, last_refresh=t)
                trajectory["i_count"] += 1
            else:
                fa_idx = INDEX_MAP[t % args.rate_gop_size]
                if mode == "P_RESET":
                    dpb["ref_feature"] = None
                    fa_idx = 3
                    trajectory.update(last_refresh=t)
                    trajectory["reset_count"] += 1
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
            psnr, psnr_y, psnr_u, psnr_v, msssim = calc_distortion(x_hat, y, u, v, args.calc_ssim)
            psnr_enc_log[t] = psnr
            cum_bits += payload_bits + outstanding_sps_bytes * 8

            rows.append({
                "schedule_id": schedule_id, "sequence_id": seq["sequence_id"],
                "frame_idx": t, "poc": t, "mode": mode, "q_index": q, "fa_idx": fa_idx,
                "payload_bits": payload_bits,
                "sps_bits": outstanding_sps_bytes * 8,
                "actual_total_bits": payload_bits + outstanding_sps_bytes * 8,
                "psnr": psnr, "psnr_y": psnr_y, "psnr_u": psnr_u, "psnr_v": psnr_v,
                "ms_ssim": msssim,
                "enc_ms": round(enc_ms, 1), "dec_ms": "",
                "ref_age": t - trajectory["last_i"],
                "reset_age": t - trajectory["last_refresh"],
                "recon_sha256": recon_hash(x_hat),
                "codec_commit": actual_commit,
                "checkpoint_hash": args.checkpoint_hash,
                "budget_remaining": "", "risk_pred": "", "action_value": "",
                "oracle_label": "",
                "prefix_actual_bits": prefix_bits if resume_meta else 0,
                "run_role": run_role,
            })
            outstanding_sps_bytes = 0
            trajectory.update(last_q=q, last_fa=fa_idx)

            if args.verbose >= 2:
                print(f"frame {t} encoded, {enc_ms/1000:.3f} s, bits: {rows[-1]['actual_total_bits']}, "
                      f"PSNR: {psnr:.4f}, MS-SSIM: {msssim:.4f}")

        src_reader.close()
        output_file.close()

        # ─── 解码循环: 完整码流从 frame 0 解码, 不读 schedule (E4) ───
        dec_helper = SPSHelper()
        input_file = bin_path.open("rb")
        src_reader = YUVReader(seq["src_path"], pic_w, pic_h)
        decoded_n = 0
        dpb_dec = None
        pending_spss = []
        p_dec_time, p_dec_count = 0.0, 0
        while decoded_n < frame_num:
            new_stream = False
            if len(pending_spss) == 0:
                header = read_header(input_file)
                if header["nal_type"] == NalType.NAL_SPS:
                    s = read_sps_remaining(input_file, header["sps_id"])
                    dec_helper.add_sps_by_id(s)
                    if args.verbose >= 2:
                        print("new sps", s)
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

            s = dec_helper.get_sps_by_id(sps_id)
            if new_stream:
                bit_stream = read_ip_remaining(input_file)
            else:
                bit_stream = None

            frame_start = time.time()
            x, y, u, v = read_src_frame(src_reader, device, args.float16)

            if header["nal_type"] == NalType.NAL_I:
                decoded = i_net.decompress(bit_stream, s)
                dpb_dec = {"ref_frame": decoded["x_hat"], "ref_feature": None,
                           "ref_mv_feature": None, "ref_y": None, "ref_mv_y": None}
                recon = decoded["x_hat"]
            else:
                if s["fa_idx"] == 3:
                    dpb_dec["ref_feature"] = None
                decoded = p_net.decompress(bit_stream, dpb_dec, s)
                dpb_dec = decoded["dpb"]
                recon = dpb_dec["ref_frame"]
                p_dec_time += decoded["decoding_time"]
                p_dec_count += 1

            recon = recon.clamp_(0, 1)
            x_hat = F.pad(recon, (-padding_l, -padding_r, -padding_t, -padding_b))
            dec_ms = (time.time() - frame_start) * 1000
            psnr, psnr_y, psnr_u, psnr_v, msssim = calc_distortion(x_hat, y, u, v, args.calc_ssim)

            # E2: 后缀帧(>=start_idx)强制双断言; 前缀帧字节级相同, 只完整解码不逐帧断言
            if decoded_n >= start_idx:
                assert abs(psnr - psnr_enc_log[decoded_n]) < 1e-9, \
                    f"frame {decoded_n}: enc/dec PSNR mismatch"
                rhash = recon_hash(x_hat)
                row_idx = decoded_n - start_idx
                assert rhash == rows[row_idx]["recon_sha256"], \
                    f"frame {decoded_n}: recon hash mismatch"
                rows[row_idx]["dec_ms"] = round(dec_ms, 1)
                rows[row_idx]["psnr"] = psnr  # 以解码侧为准回填
            decoded_n += 1

            if args.verbose >= 2 and decoded_n > start_idx:
                print(f"frame {decoded_n - 1} decoded, {dec_ms/1000:.3f} s, PSNR: {psnr:.4f}")

        input_file.close()
        src_reader.close()

    # ─── E5 记账: 完整文件级闭合 (协议: 合并后完整 .bin 必须闭合) ───
    file_bits = bin_path.stat().st_size * 8
    suffix_bits = sum(r["actual_total_bits"] for r in rows)
    if resume_meta:
        assert file_bits == prefix_bits + suffix_bits, \
            f"E5 mismatch: file={file_bits} prefix={prefix_bits} suffix={suffix_bits}"
    else:
        assert file_bits == suffix_bits, f"E5 mismatch: file={file_bits} frames={suffix_bits}"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if f.tell() == 0:
            w.writeheader()
        w.writerows(rows)

    print(f"[{run_role}] {schedule_id}{suffix}: frames {start_idx}-{frame_num - 1}, "
          f"file_bits={file_bits}")
    print(f"E2 PASS; E5 PASS (file {file_bits} == prefix {prefix_bits} + suffix {suffix_bits})")
    print(f"sha256: {sha256_file(bin_path)}")
    print(f"rows appended -> {args.output_csv}")

    if resume_meta and args.verbose >= 1:
        print(f"snapshot provenance: {args.resume_snapshot} "
              f"(pre_frame_{resume_meta['next_frame_idx']:04d}, "
              f"prefix_sha256={resume_meta['prefix_sha256'][:16]}...)")


def parse_args():
    ap = argparse.ArgumentParser(description="P0 schedule runner + P1 snapshot branching")
    ap.add_argument("--schedule", default=None, help="full-run schedule JSON")
    ap.add_argument("--branch_schedule", default=None, help="P1: branch suffix schedule JSON")
    ap.add_argument("--resume_snapshot", default=None, help="P1: snapshot dir to resume from")
    ap.add_argument("--branch_tag", default="", help="P1: output file tag for branch run")
    ap.add_argument("--model_path_i", required=True)
    ap.add_argument("--model_path_p", required=True)
    ap.add_argument("--stream_path", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--rate_gop_size", type=int, default=8, choices=[4, 8])
    ap.add_argument("--calc_ssim", action="store_true")
    ap.add_argument("--float16", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--verbose", type=int, default=2)
    ap.add_argument("--checkpoint_hash", default="")
    # P1 snapshot
    ap.add_argument("--snapshot_dir", default=None)
    ap.add_argument("--snapshot_frames", type=int, nargs="+", default=None,
                    help="save pre-frame snapshot before executing these frames")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.resume_snapshot and not args.branch_schedule:
        raise SystemExit("--resume_snapshot requires --branch_schedule")
    if not args.resume_snapshot and not args.schedule:
        raise SystemExit("either --schedule (full) or --resume_snapshot + --branch_schedule (branch)")
    run_schedule(args)