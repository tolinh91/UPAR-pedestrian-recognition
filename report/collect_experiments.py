"""Parse the leave-PETA-out proxy logs into experiments.json (per-epoch PETA score, training loss, final scores).
Logs: runs/remote_logs/*.log (vast.ai runs) and runs/*.log (laptop runs). Run from report/."""
import json, re
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "runs"; RL = R / "remote_logs"
def log(name):
    for p in (RL / f"{name}.log", R / f"{name}.log"):
        if p.exists():
            return p.read_text(errors="ignore")
    return ""
# finals given by hand where the log lacks the last.pt offset evaluation (computed with offset_proxy.py)
RUNS = [
 ("ft_PETA_dinov3", "Head linear (mốc cũ)", "Mốc", {"off06": 0.7904, "raw": 0.7785, "off10": 0.7921, "auc": 0.9076}),
 ("ft_PETA_dinov3_q", "Head query (lần 1)", "Mốc", {"off06": 0.7961, "raw": 0.7864, "off10": 0.7957, "auc": 0.9007}),
 ("q_qbase_rep", "Head query (lần 2)", "Mốc", None),
 ("q_q2", "Query 2 lớp", "Kiến trúc head", None), ("q_dec4", "Decoder 4 lớp + self-attn", "Kiến trúc head", None),
 ("q_dual", "Head kép query+linear", "Kiến trúc head", None), ("q_conv", "Head CNN (CSRA)", "Kiến trúc head", None),
 ("q_conv_l03t3", "Head CNN (CSRA, T=3)", "Kiến trúc head", None), ("q_ml4", "Token 4 tầng cuối", "Kiến trúc head", None),
 ("q_moe_tok", "MoE trên token", "Kiến trúc head", None), ("q_seq_mamba", "Mamba (SSM chọn lọc)", "Kiến trúc head", None),
 ("q_seq_attn", "Tìm tuần tự theo nhóm", "Kiến trúc head", None), ("q_qprior", "Ưu tiên vị trí bộ phận", "Kiến trúc head", None),
 ("q_qfg", "Ưu tiên token người", "Kiến trúc head", None), ("q_qprior_fg", "Vị trí + token người", "Kiến trúc head", None),
 ("q_dsheads", "Mỗi dataset 1 bộ phân loại", "Kiến trúc head", None),
 ("q_lpft", "LP-FT", "Loss / cách train", None), ("q_asl", "ASL", "Loss / cách train", None),
 ("q_la", "Logit adjustment", "Loss / cách train", None), ("q_gce", "Group CE", "Loss / cách train", None),
 ("q_rec_mae", "Decoder tái tạo (MAE)", "Loss / cách train", None), ("q_rec_attr", "Decoder tái tạo từ thuộc tính", "Loss / cách train", None),
 ("ft_PETA_dinov3_q_e3", "Query, chỉ 3 epoch", "Loss / cách train", None), ("q_noise_q", "Lọc nhãn nhiễu (1%)", "Loss / cách train", None),
 ("q_wb", "Tăng cường cân bằng trắng", "Dữ liệu / tăng cường", None), ("q_mix4", "MixStyle", "Dữ liệu / tăng cường", None),
 ("q_strongaug", "Tăng cường mạnh", "Dữ liệu / tăng cường", None), ("ft_PETA_dinov3_q_bal", "Cân bằng 3 dataset", "Dữ liệu / tăng cường", None),
 ("q_res320", "ViT-B 320×160", "Backbone / độ phân giải", None), ("q_L224", "ViT-L 224×112", "Backbone / độ phân giải", None),
 ("q_L192", "ViT-L 192×96", "Backbone / độ phân giải", None), ("q_L176", "ViT-L 176×96", "Backbone / độ phân giải", None),
 ("teacher_PETA_dinov3L", "ViT-L 256×128 (linear)", "Backbone / độ phân giải", {"off06": 0.8023, "raw": 0.7908, "off10": 0.8085, "auc": 0.9126}),
 ("ft_PETA_dinov3L_160x80", "ViT-L 160×80 (linear)", "Backbone / độ phân giải", {"off06": 0.7891, "raw": 0.7789, "off10": 0.7894, "auc": 0.8960}),
 ("student_PETA_kd", "Chưng cất logit ViT-L→B", "Backbone / độ phân giải", {"off06": 0.7921, "raw": 0.7832, "off10": 0.7937, "auc": 0.9028}),
]
out = []
for name, label, group, final in RUNS:
    t = log(name)
    if not t:
        print("missing log:", name); continue
    eps = [{"ep": int(m.group(1)), "avg": float(m.group(2))}
           for m in re.finditer(r"== ep (\d+) \(\d+s\) VAL Challenge_Avg=([0-9.]+)", t)]
    loss = [[round(int(m.group(1)) + int(m.group(2)) / int(m.group(3)), 3), float(m.group(4))]
            for m in re.finditer(r"^ep (\d+) it (\d+)/(\d+) loss ([0-9.]+)", t, re.M) if int(m.group(2)) > 0]
    if final is None:
        g = lambda pat: (re.findall(pat, t) or [None])[-1]
        final = {k: (float(v) if v else None) for k, v in {"off06": g(r"x 0\.60\s+([0-9.]+)"), "raw": g(r"x 0\.00\s+([0-9.]+)"),
                 "off10": g(r"x 1\.00\s+([0-9.]+)"), "auc": g(r"held-out PETA: AUC ([0-9.]+)")}.items()}
    out.append({"name": name, "label": label, "group": group, "epochs": eps, "loss": loss, "final": final})
json.dump(out, open(Path(__file__).parent / "experiments.json", "w"), ensure_ascii=False)
print(len(out), "experiments")
