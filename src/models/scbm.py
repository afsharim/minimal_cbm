from typing import Dict

import torch
from torch import Tensor

from .encoders import get_gaussian_encoder
from .cbm import ConceptBottleneckModel


class StochasticConceptBottleneckModel(ConceptBottleneckModel):

    def __init__(
            self,
            gamma: float=1.,
            cov_type: str="global",
            **kwargs
        ) -> None:

        self.gamma = gamma
        self.cov_type = cov_type
        
        super().__init__(**kwargs)

        self.has_concepts = True
        self.forward_returns = ['z', 'y_preds', 'c_preds']
        self.losses = ['task', 'concepts',  'precision', 'total']


    #==========Encoder p_θ(z|x)==========
    def set_encoder(self):
        self.encoder = get_gaussian_encoder(
            dim_z=self.dim_z, cov_type=self.cov_type, **self.cfg_encoder)

    def p_z_x(self, x: Tensor, n_samples: int) -> Tensor:
        return self.encoder(x, n_samples)


    #==========Representation Heads q_ϕ(z_j|c_j)==========
    def get_loss_precision(self, z_triang_cov, cov_not_triang=False):
        if cov_not_triang:
            prec_matrix = torch.inverse(z_triang_cov)
        else:
            z_triang_inv = torch.inverse(z_triang_cov)
            prec_matrix = torch.matmul(
                torch.transpose(z_triang_inv, dim0=1, dim1=2), z_triang_inv
            )
        prec_loss = prec_matrix.abs().sum(dim=(1, 2)) - prec_matrix.diagonal(
            offset=0, dim1=1, dim2=2
        ).abs().sum(-1)
        if prec_matrix.size(1) > 1:
            prec_loss = prec_loss / (prec_matrix.size(1) * (prec_matrix.size(1) - 1))
        return prec_loss.mean(-1)


    #==========Forward==========
    def forward(self, x: Tensor, c: Tensor, sampling: bool=False):
        n_samples = 1 if sampling else 0
        sampled_z, z_mu, z_triang_cov = self.p_z_x(x, n_samples)
        y_logits, y_preds = self.q_y_z(sampled_z)
        c_logits, c_preds = self.q_c_z(sampled_z)
        return {
            'z':            z_mu,
            'z_triang_cov': z_triang_cov,
            'y_logits':     y_logits,
            'y_preds':      y_preds,
            'c_logits':     c_logits,
            'c_preds':      c_preds,
        }
    
    def get_loss(
            self, 
            y: Tensor, 
            c: Tensor, 
            y_logits: Tensor,
            c_logits: Tensor,
            z_triang_cov: Tensor,
            **kwargs
        ) -> Dict[str, Tensor]:
        y_loss = self.get_loss_y(y, y_logits)
        c_loss = self.get_loss_c(c, c_logits)
        precision_loss = self.get_loss_precision(z_triang_cov)
        loss = y_loss + self.beta * c_loss + self.gamma * precision_loss
        return {
            'task':             y_loss,  
            'concepts':         c_loss,
            'precision':        precision_loss,    
            'total':            loss,
        }
    
    #==========Interventions==========
    def intervene(self, x: Tensor, c: Tensor):
        z, _, _ = self.p_z_x(x, n_samples=0)
        z_copy = self._vectorized_z_copy(z.clone(), c)
        y_logits, y_preds = self.q_y_z(z_copy)
        c_logits, c_preds = self.q_c_z(z_copy)
        return {
            'z':        z,
            'y_logits': y_logits,
            'y_preds':  y_preds,
            'c_logits': c_logits,
            'c_preds':  c_preds,
        }

    def intervention_base(self, x: Tensor) -> Tensor:
        z, _, _ = self.p_z_x(x, n_samples=0)
        return z