"""Success-probability field P(success | x, y) from the trials, five estimators, as in docs/flow.html.

Each estimator takes points (x, y, s) with s = 0/1 and returns a callable field evaluated on a grid:
p (the estimate), lo/hi (a 95 % interval or NaN) and support (False where the data is too thin to say
anything). fit on a subset is the same call on fewer points, which the cross-validated Brier score uses.
"""
from __future__ import annotations

import math

import numpy as np

Z95 = 1.959964
SC = 0.1  # feature scale for the logistic fits: 10 cm


def wilson(k, n):
    n = np.asarray(n, float)
    k = np.asarray(k, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        d = 1 + Z95 * Z95 / n
        c = p + Z95 * Z95 / (2 * n)
        m = Z95 * np.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n))
        lo, hi = np.clip((c - m) / d, 0, 1), np.clip((c + m) / d, 0, 1)
    return np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)


def _sigm(t):
    return 1 / (1 + np.exp(-t))


class Field:
    def __init__(self, p, lo, hi, support, note):
        self.p, self.lo, self.hi, self.support, self.note = p, lo, hi, support, note


def fit_kernel(pts, h):
    x, y, s = pts.T
    h2 = 2 * h * h

    def at(gx, gy):
        d2 = (x[None, :] - gx[:, None]) ** 2 + (y[None, :] - gy[:, None]) ** 2
        w = np.exp(-d2 / h2)
        sw, swy, sw2 = w.sum(1), (w * s).sum(1), (w * w).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(sw > 1e-12, swy / sw, np.nan)
            neff = np.where(sw2 > 0, sw * sw / sw2, 0)
        lo, hi = wilson(p * neff, neff)
        return Field(p, np.where(sw >= 1.5, lo, np.nan), np.where(sw >= 1.5, hi, np.nan), sw >= 1.5,
                     f"bandwidth h = {h * 100:.1f} cm")
    return at


def fit_knn(pts, k, area):
    x, y, s = pts.T
    k = min(k, len(pts))
    r_uniform = math.sqrt(k * area / (math.pi * len(pts)))  # radius holding k trials if spread evenly

    def at(gx, gy):
        d2 = (x[None, :] - gx[:, None]) ** 2 + (y[None, :] - gy[:, None]) ** 2
        order = np.argsort(d2, axis=1)[:, :k]
        p = s[order].mean(1)
        rk = np.sqrt(np.take_along_axis(d2, order[:, -1:], 1)[:, 0])
        lo, hi = wilson(p * k, np.full_like(p, k))
        return Field(p, lo, hi, rk <= 2 * r_uniform,
                     f"k = {k}. Masked where the k nearest trials spread more than twice the average ({2 * r_uniform * 100:.1f} cm)")
    return at


def fit_binned(pts, bins, box):
    x, y, s = pts.T
    (x0, x1), (y0, y1) = box
    cw, ch = (x1 - x0) / bins, (y1 - y0) / bins

    def cell(px, py):
        return (np.clip(np.floor((px - x0) / cw), 0, bins - 1).astype(int),
                np.clip(np.floor((py - y0) / ch), 0, bins - 1).astype(int))
    n = np.zeros((bins, bins))
    k = np.zeros((bins, bins))
    ci, cj = cell(x, y)
    np.add.at(n, (ci, cj), 1)
    np.add.at(k, (ci, cj), s)

    def at(gx, gy):
        gi, gj = cell(gx, gy)
        nn, kk = n[gi, gj], k[gi, gj]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(nn > 0, kk / nn, np.nan)
        lo, hi = wilson(kk, nn)
        return Field(p, lo, hi, nn >= 1, f"{bins} x {bins} cells of {cw * 100:.1f} x {ch * 100:.1f} cm")
    return at


def _irls(F, y, ridge=1e-3):
    m = F.shape[1]
    b = np.zeros(m)
    H = np.eye(m)
    for _ in range(40):
        p = _sigm(F @ b)
        w = p * (1 - p)
        H = (F * w[:, None]).T @ F + ridge * np.eye(m)
        g = F.T @ (y - p) - ridge * b
        try:
            d = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        b = b + d
        if np.abs(d).max() < 1e-9:
            break
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((m, m), np.nan)
    return b, cov


def fit_logistic(pts, quad):
    x, y, s = pts.T

    def feat(px, py):
        u, v = px / SC, py / SC
        return (np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v]) if quad
                else np.column_stack([np.ones_like(u), np.hypot(u, v)]))
    b, cov = _irls(feat(x, y), s)
    if quad:
        names = ["", " x", " y", " x²", " xy", " y²"]
        note = "logit p = " + "".join(f"{'' if j == 0 else (' + ' if v >= 0 else ' − ')}{abs(v):.2f}{names[j]}"
                                     for j, v in enumerate(b)) + " (x, y in 10 cm units)"
    else:
        note = f"logit p = {b[0]:.2f} {'+' if b[1] >= 0 else '−'} {abs(b[1]):.2f} r (r in 10 cm units)"
        if b[1] < 0:
            r50 = -b[0] / b[1] * SC
            r90 = (math.log(9) - b[0]) / b[1] * SC
            note += f". Even odds at r50 = {r50 * 100:.1f} cm; 90 % line at r = {r90 * 100:.1f} cm"
        else:
            note += ". Success does not fall with distance in this fit"

    def at(gx, gy):
        f = feat(gx, gy)
        eta = f @ b
        se = np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", f, cov, f)))
        return Field(_sigm(eta), _sigm(eta - Z95 * se), _sigm(eta + Z95 * se), np.ones(len(gx), bool), note)
    return at


ESTIMATORS = {
    "kernel": ("Kernel smoother", ("bandwidth (cm)", 0.5, 6.0, 2.0)),
    "knn": ("k nearest trials", ("k", 3, 60, 15)),
    "binned": ("Binned", ("bins per side", 4, 24, 16)),
    "logit_r": ("Logistic, radial", None),
    "logit_q": ("Logistic, quadratic", None),
}
FIELD_MIN = 8  # fewer trials than this and no field is drawn


def fitter(name, knob, box):
    (x0, x1), (y0, y1) = box
    area = (x1 - x0) * (y1 - y0)
    if name == "kernel":
        return lambda pts: fit_kernel(pts, knob / 100)
    if name == "knn":
        return lambda pts: fit_knn(pts, int(knob), area)
    if name == "binned":
        return lambda pts: fit_binned(pts, int(knob), box)
    return lambda pts: fit_logistic(pts, name == "logit_q")


def cv_brier(fit, pts, folds=10):
    """10-fold cross-validated Brier score with a fixed shuffle. Returns (score, trials scored)."""
    rng = np.random.default_rng(12345)
    idx = rng.permutation(len(pts))
    se, n = 0.0, 0
    for f in range(folds):
        test, train = idx[f::folds], np.concatenate([idx[g::folds] for g in range(folds) if g != f])
        if not len(train) or not len(test):
            continue
        field = fit(pts[train])(pts[test, 0], pts[test, 1])
        ok = np.isfinite(field.p)
        se += float(((field.p[ok] - pts[test, 2][ok]) ** 2).sum())
        n += int(ok.sum())
    return (se / n if n else float("nan")), n


def grid_field(fit, pts, box, g=40):
    """Evaluate a fitted field on a g x g grid of cell centres; returns a long table."""
    (x0, x1), (y0, y1) = box
    xs = x0 + (np.arange(g) + 0.5) * (x1 - x0) / g
    ys = y0 + (np.arange(g) + 0.5) * (y1 - y0) / g
    gx, gy = np.meshgrid(xs, ys)
    field = fit(pts)(gx.ravel(), gy.ravel())
    import pandas as pd
    return pd.DataFrame({"x": gx.ravel(), "y": gy.ravel(), "p": field.p, "lo": field.lo, "hi": field.hi,
                         "support": field.support}), field.note, (x1 - x0) / g, (y1 - y0) / g
