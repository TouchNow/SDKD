import torch
import torch.nn as nn
import torch.nn.functional as F


class SDDKDLoss(nn.Module):
    def __init__(self, temperature=4.0, complementary_weight=2.0):
        super(SDDKDLoss, self).__init__()
        self.temperature = temperature
        self.complementary_weight = complementary_weight

    def forward(self, logits_student, logits_teacher, target):
        if logits_student.shape != logits_teacher.shape:
            raise ValueError("student and teacher patch logits must have matching shapes")
        if logits_student.ndim != 3:
            raise ValueError("SDD patch logits must have shape [batch, classes, regions]")
        if logits_student.shape[2] == 0:
            raise ValueError("SDD patch logits must contain at least one region")
        if target.ndim != 1 or target.shape[0] != logits_student.shape[0]:
            raise ValueError("target must contain one class index per sample")

        teacher = logits_teacher.detach()
        log_student = F.log_softmax(logits_student / self.temperature, dim=1)
        probability_teacher = F.softmax(teacher / self.temperature, dim=1)
        region_kl = F.kl_div(
            log_student, probability_teacher, reduction="none"
        ).sum(dim=1) * self.temperature ** 2

        global_correct = teacher[:, :, 0].argmax(dim=1).eq(target)
        local_correct = teacher.argmax(dim=1).eq(target[:, None])
        weights = torch.ones_like(region_kl)
        complementary = local_correct.ne(global_correct[:, None])
        weights[complementary] = self.complementary_weight
        return (region_kl * weights).mean()
