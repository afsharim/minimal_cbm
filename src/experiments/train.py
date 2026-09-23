import os

import torch

from src.datasets import get_loader
from src.helpers import (
    calc_accuracy,
    calc_ece,
    calc_brier,
    calc_map,
    get_optimizer_scheduler,
    get_results_classifier
)
from .base import BaseExperiment


class TrainExperiment(BaseExperiment):

    experiment_name = "train"
    wandb_offline = False
    def __init__(
        self, 
        continue_training=False, 
        **kwargs
    ) -> None:
         
        super().__init__(**kwargs)
        if continue_training:
            self.ini_epoch = self.load_last_epoch()
        else:
            self.ini_epoch = 1

    #==========Setters==========
    def _set_loaders(self):
        self.cfg['data']['batch_size'] = self.cfg['training']['batch_size']
        self.train_loader, self.model_kwargs, _ = get_loader(
            train=True, seed=self.seed, **self.cfg['data'], return_nuisances=True)
        self.test_loader, _, _ = get_loader(
            train=False, seed=self.seed, **self.cfg['data'], return_nuisances=True)

    def _set_optimizer(self):
        self.optimizer, self.scheduler = get_optimizer_scheduler(
            params=self.model.parameters(),
            cfg_optimizer=self.cfg['training']['optimizer'], 
            cfg_scheduler=self.cfg['training']['scheduler'],
        )

    def _set_all(self):
        self._set_wandb()
        self._set_loaders()
        self._set_model()
        self._set_optimizer()
        self.to(self.device)
        if self.parallel:
            self.parallelize()

    #==========Steps==========
    def _step(self, x, y, c, backprop=False):
        x, y, c = self._prepare_inputs(x, y, c)
        if backprop:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(x=x, c=c, sampling=True)
                losses = self.model.get_loss(y=y, c=c, **outputs)
            self.optimizer.zero_grad(set_to_none=True)
            losses['total'].backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                outputs = self.model(x=x, c=c)
                losses = self.model.get_loss(y=y, c=c, **outputs)
        return {k: v.detach() for k, v in losses.items()}
    
    #==========Epochs==========
    def train_epoch(self, epoch):
        self.train()
        losses_epoch = {'losses_train/{}'.format(k): torch.zeros((), device='cuda') \
            for k in self.model.losses}
        for x, y, c, _, _ in self.train_loader:   
            losses_step = self._step(x, y, c, backprop=True)
            for k in self.model.losses:
                losses_epoch[f'losses_train/{k}'] += losses_step[k]
        self.scheduler.step()
        n = len(self.train_loader)
        losses_epoch = {k: (v / n).item() for k, v in losses_epoch.items()}
        if self.save:
            self.save_epoch(epoch)
        return losses_epoch

    def test_epoch(self):
        self.eval()
        losses_epoch = {'losses_test/{}'.format(k): 0 for k in self.model.losses}
        for x, y, c, _, _ in self.test_loader:
            losses_step = self._step(x, y, c)
            losses_epoch = {k: v + losses_step[k.split('/')[-1]] for k, v in losses_epoch.items()}
        losses_epoch = {k: v / len(self.test_loader) for k, v in losses_epoch.items()}
        return losses_epoch

    #==========Evaluation==========
    def _prepare_labels_preds(self, loader):
        # Cache keys once
        fkeys = tuple(self.model.forward_returns)
        # Lists of per-batch tensors (fast to concat later)
        labels_preds = {k: [] for k in fkeys}
        labels_preds.update({'y': [], 'c': [], 'n_task': [], 'n_nontask': []})
        self.model.eval()
        # Faster than no_grad(), removes autograd overhead entirely
        with torch.inference_mode():
            # (Optional) if you use AMP elsewhere, keep it here too:
            # with torch.cuda.amp.autocast(enabled=True):
            for x, y, c, n_task, n_nontask in loader:
                x, y, c = self._prepare_inputs(x, y, c)  # moves to GPU (non_blocking)
                out = self.model(x, c)
                # ONE copy per tensor per batch to CPU (no elementwise extend)
                labels_preds['y'].append(y.detach().to('cpu'))
                labels_preds['c'].append(c.detach().to('cpu'))
                labels_preds['n_task'].append(n_task.detach().to('cpu'))
                labels_preds['n_nontask'].append(n_nontask.detach().to('cpu'))
                for k in fkeys:
                    labels_preds[k].append(out[k].detach().to('cpu'))
        # Concatenate along batch dimension (0) — faster and correct
        for k in labels_preds:
            labels_preds[k] = torch.cat(labels_preds[k], dim=0)
        return labels_preds
    
    def evaluate(self, epoch):
        self.eval()
        train_set = self._prepare_labels_preds(self.train_loader) if self.save else None
        test_set  = self._prepare_labels_preds(self.test_loader)

        if self.save:
            torch.save(test_set, os.path.join(self.preds_dir, f"epoch_{epoch}.pth"))
            # Also persist a compact train split for post-hoc metrics
            # (CKA / disentanglement / OIS / NIS need train predictions too).
            keep = [k for k in ('z', 'c', 'c_preds', 'y', 'y_preds',
                                'n_task', 'n_nontask') if k in train_set]
            torch.save({k: train_set[k] for k in keep},
                       os.path.join(self.preds_dir, f"epoch_{epoch}_train.pth"))
        results = {}
        if self.save:
            results.update({
                'calibration_train/ece':   calc_ece(train_set['y_preds'], train_set['y']),
                'calibration_train/brier': calc_brier(train_set['y_preds'], train_set['y']),
                'accuracy_train/task':     calc_accuracy(train_set['y_preds'], train_set['y']),
            })
        results.update({
            'calibration_test/ece':    calc_ece(test_set['y_preds'], test_set['y']),
            'calibration_test/brier':  calc_brier(test_set['y_preds'], test_set['y']),
            'accuracy_test/task':      calc_accuracy(test_set['y_preds'], test_set['y']),
        })

        if getattr(self.model, "has_concepts", False):
            nC = self.model.n_concepts
            if self.save:
                tr_cp = train_set['c_preds']; tr_c = train_set['c']
            te_cp = test_set['c_preds'];     te_c = test_set['c']
            acc_tr_c = acc_te_c = 0.0
            map_tr_c = map_te_c = 0.0
            for j in range(nC):
                if self.save:
                    acc_tr_c += calc_accuracy(tr_cp[:, j], tr_c[:, j])
                    map_tr_c += calc_map(tr_cp[:, j],      tr_c[:, j])
                acc_te_c += calc_accuracy(te_cp[:, j], te_c[:, j])
                map_te_c += calc_map(te_cp[:, j],      te_c[:, j])
            if self.save:
                results['accuracy_train/concepts'] = acc_tr_c / nC
                results['map_train/concepts']      = map_tr_c / nC
            results['accuracy_test/concepts']  = acc_te_c / nC
            results['map_test/concepts']       = map_te_c / nC
        
        if self.save:
            z_tr = train_set['z'];     z_te = test_set['z']
            c_tr = train_set['c'];     c_te = test_set['c']

            # If still on CUDA, move once:
            if z_tr.is_cuda: z_tr = z_tr.detach().cpu()
            if z_te.is_cuda: z_te = z_te.detach().cpu()
            if c_tr.is_cuda: c_tr = c_tr.detach().cpu()
            if c_te.is_cuda: c_te = c_te.detach().cpu()

            x_train_joint = torch.cat((z_tr, c_tr), dim=-1)
            x_test_joint  = torch.cat((z_te, c_te), dim=-1)
            x_train_c     = c_tr
            x_test_c      = c_te

            # Labels (CPU NumPy) once
            n_task_tr = train_set['n_task'].detach().cpu()
            n_task_te = test_set['n_task'].detach().cpu()
            n_ntask_tr = train_set['n_nontask'].detach().cpu()
            n_ntask_te = test_set['n_nontask'].detach().cpu()

            # Counts & index lists
            n_task = n_task_tr.shape[-1]
            n_ntask = n_ntask_tr.shape[-1]
            total = n_task + n_ntask

            # Build per-nuisance label views to avoid slicing tensors inside the loop
            y_train_list = [n_task_tr[:, j] for j in range(n_task)] + [n_ntask_tr[:, j] for j in range(n_ntask)]
            y_test_list  = [n_task_te[:, j] for j in range(n_task)] + [n_ntask_te[:, j] for j in range(n_ntask)]
            key_is_task  = [True] * n_task + [False] * n_ntask  # boolean mask

            # Free the large per-sample buffers no longer needed before the
            # (long-running, ~1-2h on big datasets) leakage probe, so several
            # runs sharing the box don't exhaust host/GPU RAM. z_* were copied
            # into x_*_joint; x_*_c still alias c_*, and the y-lists are views
            # into n_*; those are kept. Numerically a no-op.
            import gc
            del train_set, test_set, z_tr, z_te
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            def get_acc_mi(y_tr, y_te):
                # Use precomputed features; sklearn likes NumPy
                res_joint = get_results_classifier(
                    x_train=x_train_joint, y_train=y_tr,
                    x_test=x_test_joint,   y_test=y_te,
                    device=self.device
                )
                res_conc = get_results_classifier(
                    x_train=x_train_c, y_train=y_tr,
                    x_test=x_test_c,   y_test=y_te,
                    device=self.device
                )
                return res_joint['accuracy'] - res_conc['accuracy'], res_joint['mi'] - res_conc['mi']

            # Main loop (cannot avoid training one classifier per nuisance, but keep it lean)
            acc_mi = [get_acc_mi(y_train_list[i], y_test_list[i]) for i in range(total)]

            # Aggregate with one pass
            sum_acc_task = 0.0; sum_acc_ntask = 0.0
            sum_mi_task  = 0.0; sum_mi_ntask  = 0.0
            for i, (acc, mi) in enumerate(acc_mi):
                if key_is_task[i]:
                    sum_acc_task += acc; sum_mi_task += mi
                else:
                    sum_acc_ntask += acc; sum_mi_ntask += mi

            results['accuracy_test/nuisances_task']     = (sum_acc_task / max(1, n_task))
            results['accuracy_test/nuisances_nontask']  = (sum_acc_ntask / max(1, n_ntask))
            results['mutual_inf_test/nuisances_task']   = (sum_mi_task  / max(1, n_task))
            results['mutual_inf_test/nuisances_nontask']= (sum_mi_ntask / max(1, n_ntask))

        return results

    #==========Run==========
    def run(self):
        import json
        log_path = os.path.join(self.results_dir, "train_log.jsonl")
        # Fresh run (bin/train.py always starts at epoch 1): truncate any log
        # left by a previous failed attempt so epochs are not duplicated.
        if self.ini_epoch <= 1:
            open(log_path, "w").close()
        n_epochs = self.cfg['training']['n_epochs']
        # Opt-in: run the (expensive) leakage probe + checkpoint only at the
        # final epoch. Training is unaffected (evaluate/probe are eval-only), so
        # the FINAL reported metrics are identical; this just skips the
        # intermediate probe rounds that only fed wandb curves.
        save_only_last = os.environ.get("MCBM_SAVE_ONLY_LAST", "0") == "1"
        for epoch in range(self.ini_epoch, n_epochs+1):
            self.save = (epoch == n_epochs) if save_only_last else \
                (epoch % self.cfg['training']['save_epochs'] == 0 or epoch == n_epochs)
            losses_train = self.train_epoch(epoch)
            losses_test = self.test_epoch()
            metrics_eval = self.evaluate(epoch)
            log_dict = {**losses_train, **losses_test, **metrics_eval}
            log_dict = {k: float(v) for k, v in log_dict.items()}
            self.wandb_run.log(log_dict)
            with open(log_path, "a") as f:
                f.write(json.dumps({'epoch': epoch, **log_dict}) + "\n")
            # Preserved per-checkpoint snapshot: retain the FULL metric record at
            # a fixed cadence in a distinct file per checkpoint, instead of only
            # overwriting final_metrics.json. This keeps the whole history (every
            # 10th epoch by default) rather than just the last one. Cheap metrics
            # are present every epoch; the expensive URR/leakage keys appear on
            # save-epochs only (the probe is ~1-2h on the large datasets).
            snap_every = int(os.environ.get("MCBM_SNAPSHOT_EVERY", "10"))
            if snap_every > 0 and (epoch % snap_every == 0 or epoch == n_epochs):
                snap_dir = os.path.join(self.results_dir, "metrics_history")
                os.makedirs(snap_dir, exist_ok=True)
                with open(os.path.join(snap_dir, f"epoch_{epoch:04d}.json"), "w") as f:
                    json.dump({'epoch': epoch, **log_dict}, f, indent=2)
            if self.save:
                with open(os.path.join(self.results_dir, "final_metrics.json"), "w") as f:
                    json.dump({'epoch': epoch, **log_dict}, f, indent=2)
            print("Epoch {}".format(epoch))
            print('\n'.join(["{}:\t{}".format(k, v) for k, v in log_dict.items()])+'\n')