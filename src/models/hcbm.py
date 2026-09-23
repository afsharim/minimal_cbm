from typing import Dict

import torch
from torch import Tensor

from .cbm import ConceptBottleneckModel


class HardConceptBottleneckModel(ConceptBottleneckModel):

    def __init__(self, **kwargs) -> None:
        
        super().__init__(**kwargs)


    #==========Concept Heads q_ϕ(c_j|z_j)==========
    @staticmethod
    def _hard_bernstein(p, thresh=0.5, ste=True):
        """Hard-threshold with optional straight-through estimator."""
        hard = (p > thresh).float()
        if ste:  # straight-through: forward = hard, backward = identity on p
            return hard + (p - p.detach())
        return hard

    def q_c_z(self, z: Tensor, thresh=0.5, ste=False) -> Tensor:
        c_logits, c_preds = super().q_c_z(z)
        c_hard = self._hard_bernstein(c_preds, thresh, ste=ste)
        return c_logits, c_preds, c_hard


    #==========Forward==========
    def forward(self, x: Tensor, c: Tensor, sampling: bool=False):
        z = self.p_z_x(x)
        c_logits, c_preds, c_hard = self.q_c_z(z)
        y_logits, y_preds = self.q_y_z(c_hard[:,:,0])
        return {
            'z':        z,
            'y_logits': y_logits,
            'y_preds':  y_preds,
            'c_logits': c_logits,
            'c_preds':  c_preds,
        }
    
    def get_loss(
            self, 
            y: Tensor, 
            c: Tensor, 
            y_logits: Tensor, 
            c_logits: Tensor, 
            **kwargs
        ) -> Dict[str, Tensor]:
        y_loss = self.get_loss_y(y, y_logits)
        c_loss = self.get_loss_c(c, c_logits)
        loss = y_loss + self.beta * c_loss
        return {
            'task':     y_loss,
            'concepts': c_loss,   
            'total':    loss,
        }
    
    #==========Interventions==========
    def _intervene_kj(self, c: Tensor, k: int, j: int):
        assert c.shape[1]==self.n_concepts, \
            "Length of c must be equal to the number of concepts"
        return c[k,self.idxs_z[j]]

    def _intervene_replacement(self, c: Tensor) -> Tensor:
        # Hard CBMs replace the (hard) predicted concept with the ground truth.
        return torch.nan_to_num(c, nan=0.0)

    def intervene(self, x: Tensor, c: Tensor):
        z = self.p_z_x(x)
        c_logits, c_preds, c_hard = self.q_c_z(z)
        z_copy = self._vectorized_z_copy(c_hard[:,:,0].clone(), c)
        y_logits, y_preds = self.q_y_z(z_copy)
        c_logits, c_preds, c_hard = self.q_c_z(z_copy)
        return {
            'z':        z,
            'y_logits': y_logits,
            'y_preds':  y_preds,
            'c_logits': c_logits,
            'c_preds':  c_preds,
        }

    # ---- Cached-encoder intervention (base = hard predicted concepts) ----
    def intervention_base(self, x: Tensor) -> Tensor:
        z = self.p_z_x(x)
        _, _, c_hard = self.q_c_z(z)
        return c_hard[:, :, 0]