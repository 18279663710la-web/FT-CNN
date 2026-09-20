# FT-CNN (MIT-BIH AAMI 5-class)

Protocol-faithful reproduction of the Biosensors 2026 FT-CNN for ECG arrhythmia classification.

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
