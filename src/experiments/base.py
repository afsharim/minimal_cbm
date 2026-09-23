import os
import random
from datetime import datetime

import numpy as np
import wandb
from pathlib import Path
import torch
from torch.autograd import Variable

from src.models import get_model, ModelParallel
from src.helpers import read_config, get_models_list


class BaseExperiment:

    def __init__(
        self, 
        config_file, 
        wandb_key, 
        device='cuda', 
        parallel=True,
        seed=42,
    ) -> None:

        self.config_file = config_file
        self.wandb_key = wandb_key
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.parallel = parallel
        self.seed = seed

        # Seed model init / training as well (the original code only seeds
        # the data configuration via random.seed inside the loaders).
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        root = Path(__file__).resolve().parents[2]
        config_subpath = "-".join(config_file.split("-")[:2]) if \
            config_file.split("-")[1]=="all" else config_file.split("-")[0]
        self.cfg = read_config(os.path.join(root, "configs", config_subpath, config_file))

        self.results_dir = os.path.join(root, "results", config_file, str(seed))
        #self.results_dir = os.path.join(root, "results", config_file)
        self.model_dir = os.path.join(self.results_dir, "models")
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self.preds_dir = os.path.join(self.results_dir, "predictions")
        Path(self.preds_dir).mkdir(parents=True, exist_ok=True)

        self._set_all()

    #==========Setters==========
    def _set_wandb(self):
        now = datetime.now()
        offline = self.wandb_offline or \
            os.environ.get("WANDB_MODE", "").lower() == "offline"
        if not offline:
            wandb.login(key=self.wandb_key)
        self.wandb_run = wandb.init(
            dir=self.results_dir,
            config=self.cfg,
            project='mcbm',
            group=self.experiment_name,
            mode="offline" if offline else "online",
            name="{experiment_name}_{config_file}_{seed}_{date}".format(
                experiment_name=self.experiment_name,
                config_file=self.config_file,
                seed=self.seed,
                date=now.strftime("%m-%d_%H:%M"),
            ),
        )

    def _set_model(self):
        self.model = get_model(**self.model_kwargs, **self.cfg['model'])

    def _set_all(self):
        raise NotImplementedError

    #==========Steps==========
    def _prepare_inputs(self, x, y, c):
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        if torch.is_tensor(c):
            c = c.to(self.device, non_blocking=True)
        return x, y, c

    #==========Save and Load==========
    def save_epoch(self, epoch):
        epoch_path = os.path.join(self.model_dir, "epoch_"+str(epoch)+".pt")
        model_state_dict = self.model.state_dict()
        checkpoint = {
            'epoch':    epoch,
            'model':    model_state_dict,
        }
        torch.save(checkpoint, epoch_path)

    def load_epoch(self, epoch):
        previous_models = get_models_list(self.model_dir, 'epoch_')
        assert 'epoch_'+str(epoch)+'.pt' in previous_models, "Selected epoch is not available"
        checkpoint = torch.load(
            os.path.join(self.model_dir, 'epoch_'+str(epoch)+'.pt'),
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model'])

    def load_last_epoch(self, restart=False):
        previous_models = get_models_list(self.model_dir, 'epoch_')
        if len(previous_models)>0 and not restart:
            print(previous_models[-1])
            checkpoint = torch.load(
                os.path.join(self.model_dir, previous_models[-1]),
                map_location=self.device
            )
            self.model.load_state_dict(checkpoint['model'])
            return checkpoint['epoch']
        else:
            return 0

    #==========Miscellanea==========
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device=None, dtype=None):
        self.model.to(device, dtype)

    def parallelize(self):
        self.model = ModelParallel(self.model)