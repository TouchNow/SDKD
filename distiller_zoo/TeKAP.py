import torch
import torch.nn as nn
import torch.nn.functional as F


class TeKAPLoss(nn.Module):
    def __init__(self, num_teachers=3, noise_weight=0.1,
                 teacher_weight=0.9, aug_weight=0.8):
        super(TeKAPLoss, self).__init__()
        self.num_teachers = num_teachers
        self.noise_weight = noise_weight
        self.teacher_weight = teacher_weight
        self.aug_weight = aug_weight

    @staticmethod
    def _kl(logits_student, logits_teacher, temperature):
        return F.kl_div(
            F.log_softmax(logits_student / temperature, dim=1),
            F.softmax(logits_teacher / temperature, dim=1),
            reduction='batchmean',
        ) * temperature * temperature

    def forward(self, logits_student, logits_teacher, temperature):
        teacher = logits_teacher.detach()
        temperature = torch.as_tensor(
            temperature, device=logits_student.device, dtype=logits_student.dtype
        )
        loss_original = self._kl(logits_student, teacher, temperature)
        loss_augmented = logits_student.new_zeros(())
        for _ in range(self.num_teachers):
            noise = torch.randn_like(teacher)
            augmented_teacher = (
                self.noise_weight * noise + self.teacher_weight * teacher
            )
            loss_augmented = loss_augmented + self._kl(
                logits_student, augmented_teacher, temperature
            )
        return loss_original + self.aug_weight * loss_augmented
