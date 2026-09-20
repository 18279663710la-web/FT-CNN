# FT-CNN (MIT-BIH AAMI 5-class)

Protocol-faithful reproduction of the Biosensors 2026 FT-CNN for ECG arrhythmia classification.

Morphology branch (single beat 187 pts) + RR-interval branch (4-dim) → 5-class softmax (N / S / V / F / Q).

## 1. 模型结构 (Model Architecture)

整体为双输入、单输出的前馈 1D-CNN，无残差、无注意力、无 LSTM：

```text
beat (187,1) ──► Conv1D(32,k=5) ─► BN ─► ReLU ─► MaxPool(2) ──┐
                                                              ├─► Concat(128+4=132) ─► FC256+BN+ReLU ─► FC128+Dropout+ReLU ─► FC64+ReLU ─► Dense(5) ─► Softmax
rr (4,) ─────────────────────────────────────────────────────┘
                        ▲
                        │
              Block3: Conv1D(128,k=3) ─► BN ─► LeakyReLU(α=0.01) ─► GAP(128,)
                        ▲
                        │
              Block2: Conv1D(64,k=3) ─► SpatialDropout(0.25) ─► PReLU ─► AvgPool(2)
```

对应代码：`src/models/ft_cnn.py::build_ft_cnn()`。

- Block1 (形态浅层特征)：大核 `k=5` 抓 QRS 主波，`MaxPool` 保留尖峰。
- Block2 (形态中层特征)：`k=3` + `SpatialDropout1D(0.25)` + 可学习 `PReLU` + `AvgPool` 平滑。
- Block3 (形态高层特征)：`128` 通道 `k=3` + `BN` + `LeakyReLU(α=0.01)` + `GAP` 压缩为 `128` 维向量 `f_cnn`。
- 融合层：`fused = concat(f_cnn(128), rr_in(4)) = 132` 维。
- 分类头：`256 → 128 → 64 → 5`，仅第一层后有 `BN`，仅第二层后有 `Dropout(0.5)`。

## 2. 网络细节参数 (Network Details)

默认超参：`conv_l2=1e-4, dense_l2=1e-4, spatial_dropout=0.25, dense_dropout=0.5, leaky_relu_alpha=0.01, temperature=1.0`。

卷积初始化 `he_normal`，全连接 `he_normal`（末层 `glorot_uniform`），卷积 `padding="same"`，池化 `pool_size=2, strides=2`，总参数量约 107k。

| 层数 | 类型 | 输出大小 | 步长 | 卷积核 / 备注 |
|------|------|----------|------|---------------|
| 1 | 输入 beat | (None,187,1) | - | 单心拍，R 前 93 + R 后 94 |
| 2 | 输入 rr | (None,4) | - | pre_RR, post_RR, local_avg(10), prematurity |
| 3 | 卷积层 Conv1 | (None,187,32) | 1 | 5×1, same, he_normal, L2 1e-4 |
| 4 | 批归一化 BN1 | (None,187,32) | - | momentum 0.99, eps 1e-3 |
| 5 | 激活 ReLU1 | (None,187,32) | - | ReLU |
| 6 | 最大池化 MaxPool1 | (None,93,32) | 2 | pool 2, stride 2 |
| 7 | 卷积层 Conv2 | (None,93,64) | 1 | 3×1, same, he_normal, L2 1e-4 |
| 8 | 空间 Dropout | (None,93,64) | - | SpatialDropout1D 0.25 |
| 9 | 激活 PReLU2 | (None,93,64) | - | PReLU (per-channel 可学习) |
| 10 | 平均池化 AvgPool2 | (None,46,64) | 2 | pool 2, stride 2 |
| 11 | 卷积层 Conv3 | (None,46,128) | 1 | 3×1, same, he_normal, L2 1e-4 |
| 12 | 批归一化 BN3 | (None,46,128) | - | momentum 0.99, eps 1e-3 |
| 13 | 激活 LeakyReLU3 | (None,46,128) | - | α=0.01 |
| 14 | 全局平均池化 GAP | (None,128) | - | 时间维平均，`f_cnn` |
| 15 | 融合 Concat | (None,132) | - | `concat(f_cnn 128, rr 4)` |
| 16 | FC1 256 + BN + ReLU | (None,256) | - | he_normal, L2 1e-4, BN+ReLU |
| 17 | FC2 128 + Dropout + ReLU | (None,128) | - | he_normal, L2 1e-4, Dropout 0.5 |
| 18 | FC3 64 + ReLU | (None,64) | - | he_normal, L2 1e-4 |
| 19 | Logits Dense | (None,5) | - | glorot_uniform, 无正则, temperature=1.0 时直通 |
| 20 | Softmax | (None,5) | - | N / S / V / F / Q |


## Setup

```bash
cd FT-CNN
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Edit `config/default.yaml` → `paths.data_root` for AutoDL.

## Local smoke

```bash
pytest -q
python scripts/smoke_test.py
```

## AutoDL full run

```bash
python scripts/build_cache.py
python -m src.train
python -m src.eval --checkpoint artifacts/checkpoints/best.h5
```

See `REPRO_NOTES.md` for paper gaps and metric deltas.

