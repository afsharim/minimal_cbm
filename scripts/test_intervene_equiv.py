"""Verify vectorized interventions == original Python double-loop.

For each intervenable model we build a tiny instance, then compare
model.intervene(x, c)['y_preds'] against a reference that reproduces the
original loop using the model's unchanged _intervene_kj.
"""
import sys
import torch

sys.path.insert(0, "/research/hal-afsharim/minimal_cbm")
from src.models import get_model

torch.manual_seed(0)

BASE = dict(
    n_concepts=6, dim_y=4, dim_c=1, continuous_y=False, continuous_c=False,
    encoder={'arch': 'mlp', 'latent_dim': 6, 'hidden_dims': [16]},
    hidden_dims_y=8, ch_in=3, image_size=8, imbalance_ratio=None,
)


def ref_loop(model, x, c, hard=False, gaussian=False):
    if gaussian:
        z, _, _ = model.p_z_x(x, n_samples=0)
    else:
        z = model.p_z_x(x)
    if hard:
        _, _, c_hard = model.q_c_z(z)
        z_base = c_hard[:, :, 0].clone()
    else:
        z_base = z.clone()
    for k in range(x.shape[0]):
        for j in range(model.n_concepts):
            if not torch.isnan(c[k, model.idxs_c[j]]):
                z_base[k, model.idxs_z[j]] = model._intervene_kj(c, k, j)
    y_logits, y_preds = model.q_y_z(z_base)
    return y_preds


def make_c(n, k):
    c = (torch.rand(n, k) > 0.5).float()
    # randomly mask ~half the entries with NaN (not intervened)
    mask = torch.rand(n, k) > 0.5
    c[mask] = float('nan')
    return c


CASES = [
    ("cbm", dict(model_type='cbm', hidden_dims_c=None, beta=0.1), False, False),
    ("hcbm", dict(model_type='hcbm', hidden_dims_c=None, beta=0.1), True, False),
    ("mcbm", dict(model_type='mcbm', hidden_dims_c=[3], beta=0.1,
                  hidden_dims_z=None, var_z=1.0, gamma=0.1), False, False),
    ("scbm", dict(model_type='scbm', hidden_dims_c=None, beta=0.1, gamma=0.01),
     False, True),
    ("shcbm", dict(model_type='shcbm', hidden_dims_c=None, beta=0.1, gamma=0.01),
     True, True),
    # arhcbm uses Linear (autoregressive) heads: hidden_dims_c=[] like the real
    # configs (None -> Identity heads would break the autoregressive concat).
    ("arhcbm", dict(model_type='arhcbm', hidden_dims_c=[], beta=0.1), True, False),
]

n = 8
K = BASE['n_concepts']
all_ok = True
for name, extra, hard, gaussian in CASES:
    model = get_model(**BASE, **extra)
    model.eval()
    # Bypass the encoder: return a fixed z (or Gaussian tuple) so the test
    # isolates the intervention logic. dim of z is sum(dim_z).
    dimz = sum(model.dim_z)
    fixed_z = torch.randn(n, dimz)
    if gaussian:
        cov = torch.eye(dimz).unsqueeze(0).repeat(n, 1, 1)
        model.p_z_x = lambda x, n_samples=0, _z=fixed_z, _c=cov: (_z, _z, _c)
    else:
        model.p_z_x = lambda x, _z=fixed_z: _z
    x = torch.randn(n, 3, 8, 8)
    c = make_c(n, K)
    if name in ("cbm", "scbm", "shcbm", "hcbm", "arhcbm"):
        # need quantiles for CBM/SCBM; harmless for hard models (unused)
        zs = torch.randn(100, BASE['n_concepts'])
        cs = (torch.rand(100, BASE['n_concepts']) > 0.5).float()
        model.prepare_interventions(zs, cs)
    with torch.no_grad():
        y_new = model.intervene(x, c)['y_preds']
        y_ref = ref_loop(model, x, c, hard=hard, gaussian=gaussian)
        # cached-base path (what InterveneExperiment now uses)
        base = model.intervention_base(x)
        y_cache = model.intervention_predict(base, c)
    diff = (y_new - y_ref).abs().max().item()
    diff_cache = (y_cache - y_ref).abs().max().item()
    ok = diff < 1e-5 and diff_cache < 1e-5
    all_ok &= ok
    print(f"[{name}] max|loop-vec|={diff:.2e} max|loop-cache|={diff_cache:.2e} "
          f"-> {'PASS' if ok else 'FAIL'}")

print("\nALL PASS" if all_ok else "\nSOME FAILED")
sys.exit(0 if all_ok else 1)
