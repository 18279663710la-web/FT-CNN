# Reproduction notes

## Paper gaps (defaults)

| Gap | Default used |
|-----|----------------|
| Lead selection | Prefer MLII / II / V5 / V2 |
| Pool kernel | 2 (stride 2) |
| Conv padding | same |
| Softmax temperature | 1.0 (disabled) |
| RR branch MLP | none (concat only) |
| Local-RR window N | 10 |
| Val patients | 223, 230 (from train list) |
| Focal+CE formula | `α_t (1-p_t)^γ * CE` |

## Phase 1 (2026-09-17) — full DS2 test (n=49833)

| Exp | Setting | Acc | Macro-F1 | N / S / V / F F1 |
|-----|---------|-----|----------|------------------|
| A | 10k/class + CE | 75.48% | 0.37 | 0.85 / 0.23 / 0.75 / 0.01 |
| B | 10k/class + Focal γ=2 | 75.81% | 0.43 | 0.85 / 0.32 / 0.91 / 0.04 |
| C | mild aug (→5k) + CE | **79.66%** | 0.40 | 0.88 / 0.23 / 0.89 / 0.01 |
| Paper | — | **98.51%** | 0.88 | 0.99 / 0.79 / 0.95 / 0.70 |

### Takeaways

1. **Best Acc = Exp C (~80%)**, still −19 pt vs paper.
2. Focal (B) helps S/V/F slightly but Acc flat; worsens N→F dumps (8518 vs A 4962).
3. Milder balance helps Acc via N recall (77%→82%) — prior mismatch is the Acc killer.
4. **V is learnable** (F1 0.75–0.91); **S/F/Q are not** at paper level under inter-patient DS2.
5. Paper Table 3 (~11k beats) ≠ standard DS2 (~50k) — evaluation protocol may differ.

## CNN+LSTM context (method A, 2026-09-18)

- Spec: `docs/superpowers/specs/2026-09-18-cnn-lstm-context-design.md`
- Config: `config/cnn_lstm.yaml` → cache `cache_ctx5`, artifacts `artifacts/cnn_lstm_ctx5`
- Input: 5 consecutive beats (center±2), shared FT-CNN morph encoder + per-step RR(4) → LSTM → center label
- Build: `python scripts/build_cache.py --config config/cnn_lstm.yaml --aug-classes 1,2,3,4`
- Train: `python -m src.train --config config/cnn_lstm.yaml`
- Result (DS2 ctx): Acc **87.83%**, S F1 **0.47** (vs Exp C Acc 79.7% / S F1 0.23)

## CNN+LSTM v2 (2026-09-18) — in progress on AutoDL

- Spec: `docs/superpowers/specs/2026-09-18-cnn-lstm-v2-design.md`
- Config: `config/cnn_lstm_v2.yaml`
- Changes: SVDB fill S/V→5k (no F/Q morph aug); attention + center CNN skip
- Cache train counts: N 41470 / S 5000 / V 5000 / F 400 / Q 8
- Artifacts: `artifacts/cnn_lstm_v2/`
