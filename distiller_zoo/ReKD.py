import torch
import torch.nn as nn
import torch.nn.functional as F


class ReKDLoss(nn.Module):
    def __init__(self):
        super(ReKDLoss, self).__init__()

    def forward(self, logits_student, logits_teacher, k, temperature, alpha, beta):
        self._validate_k(k, logits_student.size(1))
        T = self._temperature(logits_student, temperature)
        head_mask = self.head_mask(logits_student, k)
        tail_mask = 1.0 - head_mask

        tisd_loss = self._masked_kl(logits_student, logits_teacher, head_mask, T)
        ntid_loss = self._masked_kl(logits_student, logits_teacher, tail_mask, T)
        loss = alpha * tisd_loss + beta * ntid_loss
        stats = {
            "tisd": tisd_loss.detach(),
            "ntid": ntid_loss.detach(),
        }
        return loss, stats

    def head_mask(self, logits_student, k):
        self._validate_k(k, logits_student.size(1))
        head_idx = logits_student.detach().topk(k=int(k), dim=1).indices
        return torch.zeros_like(logits_student).scatter_(1, head_idx, 1.0)

    def _masked_kl(self, logits_student, logits_teacher, mask, temperature):
        masked_logits_student = logits_student / temperature - 1000.0 * (1.0 - mask)
        masked_logits_teacher = logits_teacher / temperature - 1000.0 * (1.0 - mask)
        return (
            F.kl_div(
                F.log_softmax(masked_logits_student, dim=1),
                F.softmax(masked_logits_teacher, dim=1),
                reduction="sum",
            )
            * (temperature ** 2) / logits_student.size(0)
        )

    def _temperature(self, logits_student, temperature):
        if torch.is_tensor(temperature):
            return temperature.to(device=logits_student.device, dtype=logits_student.dtype)
        return logits_student.new_tensor(float(temperature))

    def _validate_k(self, k, num_classes):
        if int(k) != k or k < 1 or k >= num_classes:
            raise ValueError("k must be an integer in [1, num_classes - 1]")
