import numpy as np
from pathlib import Path


def test_train_one_epoch_on_synthetic(tmp_path):
    """Tiny synthetic cache → one training epoch writes a keras checkpoint."""
    import tensorflow as tf

    from src.models.ft_cnn import build_ft_cnn
    from src.models.losses import categorical_focal_loss
    from src.train import class_weights_from_labels, make_datasets

    n = 64
    beats = np.random.randn(n, 187, 1).astype(np.float32)
    rr = np.random.randn(n, 4).astype(np.float32)
    labels = np.random.randint(0, 5, size=n)

    weights = class_weights_from_labels(labels)
    model = build_ft_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=categorical_focal_loss(gamma=2.0, class_weight=weights),
        metrics=["accuracy"],
    )
    ds = make_datasets(beats, labels, rr, batch_size=16, shuffle=True)
    model.fit(ds, epochs=1, verbose=0)

    out = tmp_path / "toy.h5"
    model.save(out)
    assert out.is_file()


def test_train_one_epoch_cnn_lstm_context(tmp_path):
    import tensorflow as tf

    from src.models.cnn_lstm import build_cnn_lstm
    from src.models.losses import categorical_focal_loss
    from src.train import make_datasets

    n = 64
    beats = np.random.randn(n, 5, 187, 1).astype(np.float32)
    rr = np.random.randn(n, 5, 4).astype(np.float32)
    labels = np.random.randint(0, 5, size=n)

    model = build_cnn_lstm()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=categorical_focal_loss(gamma=0.0, class_weight=None),
        metrics=["accuracy"],
    )
    ds = make_datasets(beats, labels, rr, batch_size=16, shuffle=True)
    model.fit(ds, epochs=1, verbose=0)
    out = tmp_path / "toy_ctx.h5"
    model.save(out)
    assert out.is_file()
