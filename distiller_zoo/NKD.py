import torch
import torch.nn as nn
import torch.nn.functional as F


class NKDLoss(nn.Module):
    def __init__(self, temperature=3.0, gamma=1.5):
        super(NKDLoss, self).__init__()
        self.temperature = temperature
        self.gamma = gamma

    def forward(self, logits_student, logits_teacher, target):
        if target.ndim > 1:
            label = target.argmax(dim=1, keepdim=True)
        else:
            label = target.reshape(-1, 1)

        log_probability_student = F.log_softmax(logits_student, dim=1)
        probability_teacher = F.softmax(logits_teacher, dim=1)
        target_loss = -(
            probability_teacher.gather(1, label).detach()
            * log_probability_student.gather(1, label)
        ).mean()

        non_target_mask = torch.ones_like(logits_student, dtype=torch.bool)
        non_target_mask.scatter_(1, label, False)
        batch_size = logits_student.shape[0]
        student_non_target = logits_student[non_target_mask].reshape(batch_size, -1)
        teacher_non_target = logits_teacher[non_target_mask].reshape(batch_size, -1)

        log_probability_student_non_target = F.log_softmax(
            student_non_target / self.temperature, dim=1
        )
        probability_teacher_non_target = F.softmax(
            teacher_non_target / self.temperature, dim=1
        )
        non_target_loss = -(
            probability_teacher_non_target * log_probability_student_non_target
        ).sum(dim=1).mean()
        non_target_loss = self.gamma * (self.temperature ** 2) * non_target_loss
        return target_loss + non_target_loss
