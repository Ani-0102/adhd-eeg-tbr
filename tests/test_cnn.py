import numpy as np
import torch

from src.cnn import SmallCNN, average_per_subject, train, window_probabilities


def test_cnn_gives_one_logit_per_window():
    out = SmallCNN()(torch.zeros(4, 19, 512))
    assert out.shape == (4,)


def test_average_per_subject_follows_requested_order():
    owners = np.array(["b", "a", "a", "b", "a"])
    probs = np.array([0.2, 0.4, 0.8, 0.6, 0.6])
    result = average_per_subject(probs, owners, np.array(["a", "b"]))
    assert np.allclose(result, [0.6, 0.4])


def test_cnn_learns_an_obvious_pattern():
    # class 1 windows carry a strong 10 Hz wave; class 0 windows are noise
    rng = np.random.default_rng(0)
    t = np.arange(512) / 128
    y = np.array([0, 1] * 100)
    x = rng.standard_normal((200, 19, 512)).astype(np.float32)
    x[y == 1] += (3 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)
    torch.manual_seed(0)
    model = train(x, y)
    probs = window_probabilities(model, x)
    assert probs[y == 1].mean() > probs[y == 0].mean() + 0.3
