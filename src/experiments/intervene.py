import random

import torch

from src.datasets import get_loader
from src.helpers import calc_accuracy
from .base import BaseExperiment


class InterveneExperiment(BaseExperiment):

    experiment_name = "intervene"
    wandb_offline = False

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)
        self.load_last_epoch()

    #==========Setters==========
    def _set_loaders(self):
        self.cfg['data']['batch_size'] = self.cfg['training']['batch_size']
        self.train_loader, self.model_kwargs, self.concepts_groups = get_loader(
            train=True, seed=self.seed, **self.cfg['data'])
        self.test_loader, _, _ = get_loader(
            train=False, seed=self.seed, intervention_stage=True, **self.cfg['data'])
        self.concepts = [c for group in self.concepts_groups for c in group]

    def _set_all(self):
        self._set_wandb()
        self._set_loaders()
        self._set_model()
        self.to(self.device)

    #==========Prepare intervention procedure==========
    def _prepare_representations_concepts(self, loader):
        cs, c_preds, zs = [], [], []
        for x, y, c in loader:
            with torch.no_grad():
                x, y, c = self._prepare_inputs(x, y, c)
                output = self.model(x, c)
                cs.extend(c.cpu())
                c_preds.extend(output['c_preds'].cpu())
                zs.extend(output['z'].cpu())
        return torch.stack(cs), torch.stack(c_preds), torch.stack(zs)
    

    def _prepare_interventions(self):
        cs, c_preds, zs = self._prepare_representations_concepts(self.train_loader)
        c_preds_groups = [[] for _ in range(len(self.concepts_groups))]
        for i, g in enumerate(self.concepts_groups):
            for c in g:
                c_preds_groups[i].extend(c_preds[:,self.concepts.index(c)])
        c_preds_groups = [torch.stack(g) for g in c_preds_groups]
        groups_conf = [torch.abs(0.5-i).mean() for i in c_preds_groups]
        self.groups_sorted = sorted(range(len(groups_conf)), key=lambda i: groups_conf[i], reverse=False)
        self.model.prepare_interventions(zs, cs)


    #==========Evaluation==========
    def _mask_concepts(self, c, n_int_groups, iter, random_mask=False):
        if random_mask:
            random.seed(self.seed+iter)
            int_groups = random.sample(range(0, len(self.concepts_groups)), n_int_groups)
        else:
            int_groups = self.groups_sorted[:n_int_groups]
        int_concepts = [concept for i, group in enumerate(self.concepts_groups) \
            for concept in group if i in int_groups]
        int_concepts_idxs = [self.concepts.index(c) for c in int_concepts]
        c_mod = torch.empty_like(c).fill_(torch.nan)
        c_mod[:,int_concepts_idxs] = c[:,int_concepts_idxs]
        return c_mod

    def _cache_test_bases(self):
        """Encode the test set ONCE (the encoder output is identical across
        interventions); later steps only re-run the lightweight task head.
        Only the small base (z or hard concepts) and ground-truth concepts are
        kept on-device; images are not retained."""
        self._cache = []
        for x, y, c in self.test_loader:
            with torch.no_grad():
                x, y, c = self._prepare_inputs(x, y, c)
                base = self.model.intervention_base(x)
                self._cache.append((base, y.cpu(), c))

    def _prepare_labels_preds(self, loader, n_int_groups, iter):
        ys, y_preds = [], []
        for base, y, c in self._cache:
            with torch.no_grad():
                c_mod = self._mask_concepts(c, n_int_groups, iter)
                yp = self.model.intervention_predict(base, c_mod)
                ys.append(y)
                y_preds.append(yp.cpu())
        return {'y': torch.cat(ys, 0), 'y_preds': torch.cat(y_preds, 0)}

    def evaluate(self, n_int_groups, iter=0):
        test_set = self._prepare_labels_preds(self.test_loader, n_int_groups, iter)
        return 100-calc_accuracy(test_set['y_preds'], test_set['y'])

    #==========Run==========
    # Models whose intervention procedure is defined (paper Fig. 5 / Fig. 10).
    # Vanilla and CEM have no valid concept->representation intervention.
    INTERVENABLE = {"cbm", "hcbm", "arcbm", "arhcbm", "scbm", "shcbm", "mcbm"}

    def run(self):
        import json, os
        out_path = os.path.join(self.results_dir, "intervention.json")
        model_type = str(self.cfg['model']['model_type']).lower()
        self.eval()

        if model_type not in self.INTERVENABLE:
            print("Skipping interventions: model '{}' is not intervenable"
                  .format(model_type))
            with open(out_path, "w") as f:
                json.dump({'skipped': True, 'complete': True,
                           'reason': 'model not intervenable ({})'.format(model_type),
                           'errors': [], 'errors_random': []}, f, indent=2)
            return

        # From here on we do NOT swallow errors: a genuine failure (e.g. CUDA
        # OOM) must propagate so run_queue retries instead of marking complete.
        self._prepare_interventions()
        self._cache_test_bases()
        n_groups = len(self.concepts_groups)
        n_random_iters = 5           # paper averages random policy over runs

        errors = []                  # lowest-confidence policy (Fig. 5)
        errors_random = []           # random policy (Fig. 10 / CIBM TTI)
        # Include the all-groups-intervened endpoint (0..n_groups) so the
        # curve spans 0-100% of concepts (needed for AUC/NAUC-TTI).
        for n_int_groups in range(0, n_groups + 1):
            with torch.no_grad():
                err_lc = float(self.evaluate(n_int_groups))
                if n_int_groups in (0, n_groups):
                    err_rand = err_lc     # deterministic at the endpoints
                else:
                    err_rand = float(sum(
                        self.evaluate(n_int_groups, i)
                        for i in range(n_random_iters)) / n_random_iters)
                errors.append(err_lc)
                errors_random.append(err_rand)
                print("groups={} err_lowconf={:.3f} err_random={:.3f}"
                      .format(n_int_groups, err_lc, err_rand))
                self.wandb_run.log({'error': err_lc, 'error_random': err_rand})
                with open(out_path, "w") as f:
                    json.dump({
                        'skipped': False,
                        'complete': (n_int_groups == n_groups),
                        'n_groups': n_groups,
                        'policy': 'lowest_confidence + random',
                        'random_iters': n_random_iters,
                        'errors': errors,
                        'errors_random': errors_random,
                    }, f, indent=2)