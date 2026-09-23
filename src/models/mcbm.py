from typing import List, Union, Dict

import torch
from torch import nn, Tensor

from .cbm import ConceptBottleneckModel


class MinimalConceptBottleneckModel(ConceptBottleneckModel):

    def __init__(
            self,
            hidden_dims_z: Union[int, List[int], List[List[int]]],
            gamma: float=1.,
            **kwargs
        ) -> None:
        
        super().__init__(**kwargs)

        if isinstance(hidden_dims_z, int):
            hidden_dims_z = [hidden_dims_z]
        if hidden_dims_z==[] or hidden_dims_z==None or isinstance(hidden_dims_z[0], int):
            hidden_dims_z = [hidden_dims_z] * self.n_concepts

        self.hidden_dims_z = hidden_dims_z
        self.gamma = gamma
        
        self.set_heads_z()

        self.has_concepts = True
        self.forward_returns = ['z', 'y_preds', 'c_preds', 'z_preds']
        self.losses = ['task', 'concepts',  'representations', 'total']


    #==========Representation Heads q_ϕ(z_j|c_j)==========
    def set_heads_z_mlp(self):
        assert len(self.dim_z)==len(self.continuous_z)==len(self.hidden_dims_z), \
            "dim_z, continuous_z and hidden_dims_z must have the same length"
        mlp_z, act_z, loss_z = [], [], []
        for j in range(self.n_concepts):
            mlp_z_j, act_z_j = self._build_head(
                input_dim=self.dim_c[j], 
                hidden_dims=self.hidden_dims_z[j], 
                output_dim=self.dim_z[j],
                continuous=self.continuous_z[j]
            )
            loss_z_j = self._get_loss_fn(self.dim_z[j], self.continuous_z[j])
            mlp_z.append(mlp_z_j)
            act_z.append(act_z_j)
            loss_z.append(loss_z_j)
        self.mlp_z = nn.ModuleList(mlp_z)
        self.act_z = nn.ModuleList(act_z)
        self.loss_z = nn.ModuleList(loss_z)

    def set_heads_z(self):
        assert len(self.dim_z)==len(self.continuous_z)==len(self.hidden_dims_z), \
            "dim_z, continuous_z and hidden_dims_z must have the same length"
        prior_z, loss_z = [], []
        for j in range(self.n_concepts):
            prior_z_j = nn.Linear(self.dim_c[j], self.dim_z[j])
            loss_z_j = self._get_loss_fn(self.dim_z[j], self.continuous_z[j])
            prior_z.append(prior_z_j)
            loss_z.append(loss_z_j)
        self.prior_z = nn.ModuleList(prior_z)
        self.loss_z = nn.ModuleList(loss_z)

    def q_z_c(self, c: Tensor) -> Tensor:
        """
        z_logits = torch.stack([
            self.mlp_z[j](c[:,self.idxs_c[j]]) for j in range(self.n_concepts)
        ]).permute(1, 0, 2)
        z_preds = torch.stack([
            self.act_z[j](z_logits[:,self.idxs_z[j]]) for j in range(self.n_concepts)
        ])[:,:,:,0].permute(1, 0, 2)
        """
        z_logits = torch.unsqueeze(6 * c - 3, -1)
        z_preds = z_logits
        return z_logits, z_preds

    def get_loss_z(self, z: Tensor, z_logits: Tensor, c: Tensor) -> Tensor:
        loss_z = None
        for j in range(self.n_concepts):
            #loss_z_j = self.loss_z[j](z_logits[:,self.idxs_z[j]], z[:,self.idxs_z[j]])
            #loss_z_j = ((0.8 *c[:,j] + 0.1) * ((z_logits[:,self.idxs_z[j],0]-z[:,self.idxs_z[j]])[:,0])**2).mean()
            loss_z_j = 0.2 * (((z_logits[:,self.idxs_z[j],0]-z[:,self.idxs_z[j]])[:,0])**2).mean()
            if loss_z is None:
                loss_z = loss_z_j
            else:
                loss_z.add_(loss_z_j)
        return loss_z

    #==========Forward==========
    def forward(self, x: Tensor, c: Tensor, sampling: bool=False):
        z = self.p_z_x(x)
        sampled_z = z + self.var_z * torch.randn_like(z) if sampling else z
        y_logits, y_preds = self.q_y_z(sampled_z)
        c_logits, c_preds = self.q_c_z(sampled_z)
        z_logits, z_preds = self.q_z_c(c)
        return {
            'z':        z,
            'y_logits': y_logits,
            'y_preds':  y_preds,
            'c_logits': c_logits,
            'c_preds':  c_preds,
            'z_logits': z_logits,
            'z_preds':  z_preds,
        }
    
    def get_loss(
            self, 
            y: Tensor, 
            c: Tensor, 
            z: Tensor,
            y_logits: Tensor,
            c_logits: Tensor,
            z_logits: Tensor,
            **kwargs
        ) -> Dict[str, Tensor]:
        y_loss = self.get_loss_y(y, y_logits)
        c_loss = self.get_loss_c(c, c_logits)
        z_loss = self.get_loss_z(z, z_logits, c)
        loss = y_loss + self.beta * c_loss + self.gamma * z_loss
        return {
            'task':             y_loss,  
            'concepts':         c_loss,
            'representations':  z_loss,    
            'total':            loss,
        }
    
    #==========Interventions==========
    def _intervene_kj(self, c: Tensor, k: int, j: int):
        return torch.unsqueeze(6 * c[k,self.idxs_c[j]] - 3, -1)
        #z_j_logits = self.mlp_z[j](c[k,self.idxs_c[j]].unsqueeze(0))
        #return self.act_z[j](z_j_logits)

    def _intervene_replacement(self, c: Tensor) -> Tensor:
        # MCBM interventions use the learned prior mapping q(z_j|c_j), which in
        # this implementation is the deterministic map z_j = 6*c_j - 3.
        return 6 * torch.nan_to_num(c, nan=0.0) - 3