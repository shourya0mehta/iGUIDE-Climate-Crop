"""First-order MAML meta-training for the LOCO crop-yield evaluation.

Per-fold usage (see training/loco_eval.py): the caller builds `region_tensors`
from the five training regions of the fold only, with features scaled and
targets z-normalized using statistics fit on those same training regions.
Nothing in this module ever sees the held-out region until `adapt_and_predict`
is called with the held-out support set at meta-test time.
"""
import numpy as np
import torch
import torch.nn as nn

import learn2learn as l2l


def sample_task(X, y, support_size, query_size, rng):
    """Draw one (support, query) episode from a region without replacement.

    Support and query are always disjoint within an episode. When the region
    pool is smaller than support_size + query_size (never the case for this
    dataset, min region n=99 >= 64), fall back to sampling with replacement.
    """
    n = X.shape[0]
    take = support_size + query_size
    if n >= take:
        idx = rng.choice(n, size=take, replace=False)
    else:
        idx = rng.choice(n, size=take, replace=True)
    idx = torch.from_numpy(np.ascontiguousarray(idx))
    s, q = idx[:support_size], idx[support_size:]
    return X[s], y[s], X[q], y[q]


def meta_train(model, region_tensors, epochs=200, inner_lr=1e-3, outer_lr=3e-4,
               support_size=32, query_size=32, inner_steps=3, grad_clip=1.0,
               first_order=True, rng=None, log_every=100, label=""):
    """Meta-train `model` with first-order MAML over region tasks.

    region_tensors: dict region_name -> (X FloatTensor [n, d], y FloatTensor [n]),
        containing ONLY the training regions of the fold.
    Each epoch samples one episode per region (so every region contributes
    equally regardless of its size), averages the post-adaptation query losses,
    and takes one clipped Adam step on the meta-parameters.

    Returns the learn2learn MAML wrapper; pass it to adapt_and_predict.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    regions = sorted(region_tensors)

    for r in regions:
        n = int(region_tensors[r][1].shape[0])
        if n < 2 * (support_size + query_size):
            print("    [warn] training region '%s' has only n=%d rows for "
                  "support+query=%d episodes; draws will heavily overlap "
                  "across episodes" % (r, n, support_size + query_size),
                  flush=True)

    maml = l2l.algorithms.MAML(model, lr=inner_lr, first_order=first_order)
    opt = torch.optim.Adam(maml.parameters(), lr=outer_lr)
    loss_fn = nn.MSELoss()

    maml.train()
    for epoch in range(epochs):
        opt.zero_grad()
        meta_loss = 0.0
        for r in regions:
            X, y = region_tensors[r]
            sx, sy, qx, qy = sample_task(X, y, support_size, query_size, rng)
            learner = maml.clone()
            for _ in range(inner_steps):
                learner.adapt(loss_fn(learner(sx).reshape(-1), sy))
            meta_loss = meta_loss + loss_fn(learner(qx).reshape(-1), qy)
        meta_loss = meta_loss / len(regions)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(maml.parameters(), grad_clip)
        opt.step()
        if log_every and (epoch + 1) % log_every == 0:
            print("    %s epoch %d/%d  meta-loss %.4f"
                  % (label, epoch + 1, epochs, float(meta_loss)), flush=True)
    return maml


def adapt_and_predict(maml_model, support_X, support_y, query_X, inner_steps=10):
    """Clone the meta-trained model, adapt on the support set, predict the query set.

    Inputs may be numpy arrays or tensors; predictions are returned as a numpy
    array in the same (z-normalized) target space the model was trained in.
    """
    support_X = torch.as_tensor(support_X, dtype=torch.float32)
    support_y = torch.as_tensor(support_y, dtype=torch.float32)
    query_X = torch.as_tensor(query_X, dtype=torch.float32)

    loss_fn = nn.MSELoss()
    learner = maml_model.clone()
    learner.train()
    for _ in range(inner_steps):
        learner.adapt(loss_fn(learner(support_X).reshape(-1), support_y))
    learner.eval()
    with torch.no_grad():
        preds = learner(query_X).reshape(-1)
    return preds.numpy()
