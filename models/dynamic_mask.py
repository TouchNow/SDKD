import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicTopKMaskGenerator(nn.Module):
    def __init__(self, num_classes, hidden_dim=128, eps=1e-8):
        super(DynamicTopKMaskGenerator, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.fc1 = nn.Linear(num_classes * 3, hidden_dim)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, logit_t, logit_s, head_ratio, gumbel_tau=1.0):
        state = torch.cat([
            logit_t.detach(),
            logit_s.detach(),
            (logit_t - logit_s).abs().detach(),
        ], dim=1)
        score = self.out(self.act(self.fc1(state)))
        prob = F.softmax(score, dim=1)
        head_k = self._head_k(head_ratio)

        if self.training:
            gumbel = -torch.empty_like(score).exponential_().log()
            sampled_score = (torch.log(prob.clamp_min(self.eps)) + gumbel) / max(gumbel_tau, self.eps)
        else:
            sampled_score = prob

        head_idx = sampled_score.topk(k=head_k, dim=1).indices
        hard_mask = torch.zeros_like(score).scatter_(1, head_idx, 1.0)
        soft_mask = F.softmax(score / max(gumbel_tau, self.eps), dim=1) * head_k
        mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        return mask, score, prob

    def _head_k(self, head_ratio):
        head_k = int(math.ceil(float(head_ratio) * self.num_classes))
        return min(max(head_k, 1), self.num_classes - 1)
