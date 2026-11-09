import torch
import torch.nn as nn
import torch.nn.functional as F


class INARankLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(INARankLoss, self).__init__()
        self.eps = eps

    def forward(self, logits_student, logits_teacher, target):
        batch_size = logits_student.size(0)
        if batch_size < 2:
            zero = logits_student.sum() * 0.0
            stats = {
                "ina_rank": zero.detach(),
                "pairs": logits_student.new_tensor(0),
            }
            return zero, stats

        target_student = self._standardized_target_score(logits_student, target)
        with torch.no_grad():
            target_teacher = self._standardized_target_score(logits_teacher, target)

        row_idx, col_idx = torch.triu_indices(batch_size, batch_size, offset=1, device=logits_student.device)
        teacher_diff = target_teacher[row_idx] - target_teacher[col_idx]
        student_diff = target_student[row_idx] - target_student[col_idx]

        relation = torch.where(
            teacher_diff >= 0,
            torch.ones_like(teacher_diff),
            -torch.ones_like(teacher_diff),
        )
        margin = teacher_diff.abs()  # 教师认为两个样本难度之间的相对距离
        loss = F.relu(-relation * student_diff + margin).mean()
        stats = {
            "ina_rank": loss.detach(),
            "pairs": logits_student.new_tensor(row_idx.numel()),
        }
        return loss, stats

    def _standardized_target_score(self, logits, target):
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        normalized = (logits - mean) / std
        return normalized.gather(1, target.view(-1, 1)).squeeze(1)
