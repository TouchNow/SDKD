import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicDualKDLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(DynamicDualKDLoss, self).__init__()
        self.eps = eps

    def forward(self, logits_student, logits_teacher, mask, temperature,
                head_weight=1.0, tail_weight=1.0, mass_weight=1.0):
        T = logits_student.new_tensor(float(temperature))
        pred_student = F.softmax(logits_student / T, dim=1)
        pred_teacher = F.softmax(logits_teacher / T, dim=1)

        head_mask = mask
        tail_mask = 1.0 - mask

        head_student, head_teacher, head_mass_student, head_mass_teacher = self._masked_probs(
            pred_student, pred_teacher, head_mask
        )
        tail_student, tail_teacher, tail_mass_student, tail_mass_teacher = self._masked_probs(
            pred_student, pred_teacher, tail_mask
        )

        head_kl = self._kl(head_student, head_teacher) * (T ** 2)
        tail_kl = self._kl(tail_student, tail_teacher) * (T ** 2)

        mass_student = torch.cat([head_mass_student, tail_mass_student], dim=1)
        mass_teacher = torch.cat([head_mass_teacher, tail_mass_teacher], dim=1)
        mass_kl = self._kl(mass_student, mass_teacher) * (T ** 2)

        loss = head_weight * head_kl + tail_weight * tail_kl + mass_weight * mass_kl
        stats = {
            "head_kl": head_kl.detach(),
            "tail_kl": tail_kl.detach(),
            "mass_kl": mass_kl.detach(),
        }
        return loss, stats

    def _masked_probs(self, pred_student, pred_teacher, mask):
        mass_student = (pred_student * mask).sum(dim=1, keepdim=True).clamp_min(self.eps)
        mass_teacher = (pred_teacher * mask).sum(dim=1, keepdim=True).clamp_min(self.eps)
        masked_student = pred_student * mask / mass_student
        masked_teacher = pred_teacher * mask / mass_teacher
        return masked_student, masked_teacher, mass_student, mass_teacher

    def _kl(self, pred_student, pred_teacher):
        log_teacher = torch.log(pred_teacher.clamp_min(self.eps))
        log_student = torch.log(pred_student.clamp_min(self.eps))
        return (pred_teacher * (log_teacher - log_student)).sum(dim=1).mean()
