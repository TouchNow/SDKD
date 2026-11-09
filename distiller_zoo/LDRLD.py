import torch
import torch.nn as nn
import torch.nn.functional as F


class LDRLDLoss(nn.Module):
    def __init__(self, irw_eps=1.5, erd_delta=2.0, erd_lambda=0.05):
        super(LDRLDLoss, self).__init__()
        self.irw_eps = irw_eps
        self.erd_delta = erd_delta
        self.erd_lambda = erd_lambda

    def forward(self, logits_student, logits_teacher, depth=7, temperature=4.0,
                alpha=7.0, beta=4.0):
        self._validate_depth(depth, logits_student.size(1))
        temperature = self._temperature(logits_student, temperature)
        logits_teacher = logits_teacher.detach()

        top_idx = self.topk_indices(logits_student, depth)
        top_student = logits_student.gather(1, top_idx)
        top_teacher = logits_teacher.gather(1, top_idx)

        weighted_local = self._weighted_pairwise_kl(top_student, top_teacher, temperature)
        llki = self._kl(top_student, top_teacher, temperature)
        rntk = self._remaining_kl(logits_student, logits_teacher, top_idx, temperature)

        local = weighted_local + llki
        loss = alpha * local + beta * rntk
        stats = {
            "local": local.detach(),
            "weighted_local": weighted_local.detach(),
            "llki": llki.detach(),
            "rntk": rntk.detach(),
        }
        return loss, stats

    def topk_indices(self, logits_student, depth):
        self._validate_depth(depth, logits_student.size(1))
        return logits_student.detach().topk(depth, dim=1).indices

    def _weighted_pairwise_kl(self, top_student, top_teacher, temperature):
        depth = top_student.size(1)
        left, right = self._pair_indices(depth, top_student.device)
        pair_student = torch.stack([top_student[:, left], top_student[:, right]], dim=2)
        pair_teacher = torch.stack([top_teacher[:, left], top_teacher[:, right]], dim=2)
        pair_kl = F.kl_div(
            F.log_softmax(pair_student / temperature, dim=2),
            F.softmax(pair_teacher / temperature, dim=2),
            reduction="none",
        ).sum(dim=2)
        weights = self._adw_weights(left, right, top_student.dtype)
        return (pair_kl * weights).sum(dim=1).mean()

    def _remaining_kl(self, logits_student, logits_teacher, top_idx, temperature):
        remaining_mask = torch.ones_like(logits_student, dtype=torch.bool)
        remaining_mask.scatter_(1, top_idx, False)
        remaining_student = logits_student.masked_fill(~remaining_mask, -1000.0)
        remaining_teacher = logits_teacher.masked_fill(~remaining_mask, -1000.0)
        return self._kl(remaining_student, remaining_teacher, temperature)

    def _kl(self, logits_student, logits_teacher, temperature):
        return F.kl_div(
            F.log_softmax(logits_student / temperature, dim=1),
            F.softmax(logits_teacher / temperature, dim=1),
            reduction="batchmean",
        )

    def _adw_weights(self, left, right, dtype):
        rank_left = left.to(dtype=dtype) + 1.0
        rank_right = right.to(dtype=dtype) + 1.0
        irw = 1.0 / (torch.abs(rank_right - rank_left) + self.irw_eps)
        erd = self.erd_delta * torch.exp(-self.erd_lambda * (rank_left + rank_right))
        return irw * erd

    def _pair_indices(self, depth, device):
        left = []
        right = []
        for i in range(depth - 1):
            for j in range(i + 1, depth):
                left.append(i)
                right.append(j)
        return torch.tensor(left, device=device), torch.tensor(right, device=device)

    def _temperature(self, logits_student, temperature):
        if torch.is_tensor(temperature):
            return temperature.to(device=logits_student.device, dtype=logits_student.dtype)
        return logits_student.new_tensor(float(temperature))

    def _validate_depth(self, depth, num_classes):
        if int(depth) != depth or depth < 2 or depth >= num_classes:
            raise ValueError("depth must be an integer in [2, num_classes - 1]")
