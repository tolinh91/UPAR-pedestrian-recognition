# UPAR-pedestrian-recognition

Code cho **UPAR Challenge 2027 – Track 1: Pedestrian Attribute Recognition** (RWS @ WACV 2027,
[Codabench](https://www.codabench.org/competitions/18180/)).

Cho một ảnh crop người đi bộ, model trả về 40 xác suất cho 40 thuộc tính nhị phân. Model train trên
Market1501, PA100k và PETA, được chấm trên test ẩn từ domain chưa thấy. Bài nộp là code: server chạy
`run.py` trong container `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`, không có mạng, giới hạn 1 giờ, log bị ẩn.

Báo cáo tổng quan (bài toán, dữ liệu, metric, vấn đề cần giải quyết): [`report/track1_report.pdf`](report/track1_report.pdf).

## Kết quả hiện tại

| Model | Val (cùng domain) Avg | Test Avg | mA | Inst-F1 |
|---|---|---|---|---|
| DINOv3 ViT-B/16, 256×128, offset ×0.6, train (`results/dinov3_b16_v3`) | 0.885 | 0.7774 | 0.7786 | 0.7762 |
| Như trên nhưng train trên train + val, 8 epoch (`results/dinov3_b16_trainval_v4`) | (val nằm trong train) | **0.7775** | 0.7781 | 0.7769 |

Thêm val (cùng 3 domain) gần như không đổi điểm test: cải thiện tiếp phải đến từ khả năng tổng quát
sang domain lạ hoặc năng lực model, không đến từ thêm dữ liệu cùng domain.

## Cấu trúc

```
par/
  upar/                 package dùng chung
    registry.py         đăng ký thành phần theo tên, parse tham số K=V
    data.py             annotations, cache ảnh, augmentation, dataset
    metrics.py          mA, Instance-F1, Challenge_Avg (chấm riêng từng dataset rồi lấy trung bình)
    losses.py           LOSSES: bce, logit_adj, focal, asl + tuỳ chọn chung smooth / weighting / mask
    models/             HEADS: linear (v3), query (attention riêng từng thuộc tính) + build_model()
  train.py              train với --loss/--head chọn theo tên, --holdout (proxy domain lạ), --resume
  calibrate.py          fit offset logit từng thuộc tính (ngưỡng 0.5 bị BTC cố định), kiểm chứng 2-fold CV
  offset_lodo.py        offset fit trên 2 dataset, chấm trên dataset thứ 3
  wise.py               WiSE-FT: nội suy trọng số pretrained ↔ fine-tuned
  make_submission.py    đóng gói zip nộp bài (run.py + timm + upar.models vendored + weights fp16 + config)
  test_submission.py    giả lập ingestion trên val
  fake_ingestion.py     giả lập ingestion trong docker trên test set giả
  tests/                unit test loss và head (python tests/test_components.py)
submission_template/run.py   entry point nộp bài: tự chọn GPU/CPU, ngân sách thời gian, chống NaN/ảnh hỏng
colab/RWS2027_colab.ipynb    notebook train trên Colab (resume khi bị ngắt)
eda/eda_images.py            EDA ảnh
report/                      báo cáo LaTeX + script vẽ hình
results/                     config và offset của các bài đã nộp
```

## Cài đặt local

```bash
# 1. starter kit + dữ liệu của BTC (không nằm trong repo này)
git clone https://github.com/speckean/UPAR-Challenge-2027
cd UPAR-Challenge-2027
sed -i 's/, fuzzy=True)/)/' download_datasets.py   # gdown >= 6 đã bỏ tham số fuzzy
python download_datasets.py --data-dir ./data
cd ..

# 2. môi trường
python3 -m venv .venv
.venv/bin/pip install torch torchvision timm==1.0.30 pandas pillow matplotlib
```

Thêm loss mới: viết class kế thừa `BaseLoss` trong `upar/losses.py` và gắn `@LOSSES.register("ten")`.
Thêm head mới: class trong `upar/models/heads.py` với `@HEADS.register("ten")`, `forward(tokens, n_prefix)`.

Lần chạy đầu, `par/upar/data.py` gộp toàn bộ ảnh vào `cache/{train,val}_bytes.npz` (~0.66 GB) để DataLoader đọc nhanh.

## Train, hiệu chỉnh, đóng gói

```bash
cd par
../.venv/bin/python train.py --list                                               # loss / head có sẵn
../.venv/bin/python train.py --out ../runs/dinov3_b16 --epochs 10                 # công thức v3
../.venv/bin/python train.py --loss asl --loss-kw gamma_neg=4 clip=0.05 --out ../runs/asl
../.venv/bin/python train.py --loss logit_adj --loss-kw tau=0.5 smooth=0.05 --out ../runs/la
../.venv/bin/python train.py --head query --head-kw n_layers=1 --out ../runs/query
../.venv/bin/python train.py --train-splits train val --epochs 8 --out ../runs/trainval   # bản v4
../.venv/bin/python train.py --holdout PETA --epochs 5 --out ../runs/lodo_PETA   # proxy domain lạ
../.venv/bin/python calibrate.py ../runs/dinov3_b16                               # offset ngưỡng
../.venv/bin/python make_submission.py ../runs/dinov3_b16 --offset-scale 0.6 --name my_sub
../.venv/bin/python test_submission.py ../submissions/my_sub --n 2000              # kiểm tra trước khi nộp
```

Kiểm tra trong đúng container chấm bài (không mạng, giới hạn 2 CPU):

```bash
docker run --rm --network none --cpus=2 --memory=3g \
  -v $PWD/submissions/my_sub:/app/program:ro -v $PWD/fake_input_data:/app/input_data:ro \
  -v $PWD/par/fake_ingestion.py:/ing.py:ro -w /app/program \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime python /ing.py
```

## Colab

Upload `rws2027_bundle.zip` (code + annotations + cache ảnh, tạo bằng lệnh dưới) vào `MyDrive/RWS2027/`,
mở `colab/RWS2027_colab.ipynb` và chạy lần lượt các cell. Checkpoint lưu thẳng vào Drive.

```bash
zip -0 -r colab/rws2027_bundle.zip par/*.py par/upar submission_template/run.py \
  UPAR-Challenge-2027/data/annotations/task1 \
  UPAR-Challenge-2027/examples/task1/sample_code_submission/metadata.yaml \
  UPAR-Challenge-2027/examples/task1/sample_code_submission/assets/attribute_prior.json \
  cache/train_bytes.npz cache/val_bytes.npz
```

## Lưu ý luật thi

- Chỉ dùng dữ liệu BTC cung cấp. Model pretrain chung (ImageNet, DINOv3, CLIP…) được phép.
- Cấm mọi cách dùng ảnh test, kể cả test-time adaptation và chọn siêu tham số theo test.
- Leaderboard giữ bài nộp **mới nhất**, 3 lượt/ngày. Hạn nộp 21/10/2026 AoE.
- Dữ liệu UPAR theo giấy phép CC BY-NC-SA 3.0. Không commit ảnh hoặc cache dữ liệu vào repo.
