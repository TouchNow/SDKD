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

        row_idx, col_idx = torch.triu_indices(
            batch_size,
            batch_size,
            offset=1,
            device=logits_student.device,
        )
        teacher_diff = target_teacher[row_idx] - target_teacher[col_idx]
        student_diff = target_student[row_idx] - target_student[col_idx]

        relation = torch.where(
            teacher_diff >= 0,
            torch.ones_like(teacher_diff),
            -torch.ones_like(teacher_diff),
        )
        margin = teacher_diff.abs()
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


def probability_boundaries(probabilities):
    if probabilities.ndim != 2 or probabilities.size(1) < 2:
        raise ValueError(
            "probabilities must have shape [batch, ranks] with at least 2 ranks"
        )

    gaps = probabilities[:, :-1] - probabilities[:, 1:]
    return torch.argmax(gaps, dim=1) + 1


class AHTKDLoss(nn.Module):
    def __init__(self, prob_eps=1e-12):
        super().__init__()
        if prob_eps <= 0:
            raise ValueError("prob_eps must be > 0")
        self.prob_eps = prob_eps

    def forward(
            self,
            logits_student,
            logits_teacher,
            target,
            temperature=4.0,
            lambda_head=1.0,
            lambda_tail=1.0,
    ):
        B, C = logits_teacher.shape
        device = logits_teacher.device

        with torch.no_grad():
            target = target.reshape(-1).to(device=device, dtype=torch.long)

            teacher_logits_fp32 = logits_teacher.detach().float()

            class_indices = torch.arange(C, device=device).unsqueeze(0).expand(B, C)
            non_target_mask = class_indices != target.unsqueeze(1)
            non_target_indices = class_indices[non_target_mask].reshape(B, C - 1)
            non_target_teacher_logits = teacher_logits_fp32[
                non_target_mask
            ].reshape(B, C - 1)

            sorted_non_target_logits, non_target_order = torch.sort(
                non_target_teacher_logits, dim=1, descending=True
            )
            sorted_non_target_indices = torch.gather(
                non_target_indices, dim=1, index=non_target_order
            )
            ordered_indices = torch.cat(
                [target.unsqueeze(1), sorted_non_target_indices], dim=1
            )

            boundary_prob = F.softmax(sorted_non_target_logits, dim=1)
            boundaries = probability_boundaries(boundary_prob)

            ordered_teacher_logits_fp32 = torch.gather(
                teacher_logits_fp32, dim=1, index=ordered_indices
            )

        ordered_student_logits = torch.gather(
            logits_student, dim=1, index=ordered_indices
        )

        ordered_teacher_logits = ordered_teacher_logits_fp32.to(
            dtype=logits_student.dtype
        )

        T = logits_student.new_tensor(float(temperature))
        scaled_teacher_logits = ordered_teacher_logits / T
        scaled_student_logits = ordered_student_logits / T

        t_log_prob = F.log_softmax(scaled_teacher_logits, dim=1)
        s_log_prob = F.log_softmax(scaled_student_logits, dim=1)
        t_prob = t_log_prob.exp()
        s_prob = s_log_prob.exp()

        non_target_rank = torch.arange(C - 1, device=device).unsqueeze(0)
        non_target_head_mask = non_target_rank < boundaries.unsqueeze(1)
        head_mask = torch.cat(
            [
                torch.ones((B, 1), dtype=torch.bool, device=device),
                non_target_head_mask,
            ],
            dim=1,
        )
        tail_mask = ~head_mask

        zero = torch.zeros_like(t_prob)

        head_terms = torch.where(
            head_mask,
            t_prob * (t_log_prob - s_log_prob),
            zero,
        )

        t_tail_mass = (t_prob * tail_mask).sum(dim=1).clamp_min(self.prob_eps)
        s_tail_mass = (s_prob * tail_mask).sum(dim=1).clamp_min(self.prob_eps)

        head_kl_per_sample = (
                head_terms.sum(dim=1)
                + t_tail_mass * (torch.log(t_tail_mass) - torch.log(s_tail_mass))
        )

        head_loss = head_kl_per_sample.mean() * T * T

        mask_value = torch.finfo(scaled_teacher_logits.dtype).min

        tail_teacher_logits = scaled_teacher_logits.masked_fill(
            head_mask, mask_value
        )
        tail_student_logits = scaled_student_logits.masked_fill(
            head_mask, mask_value
        )

        tail_log_prob = F.log_softmax(tail_teacher_logits, dim=1)
        student_tail_log_prob = F.log_softmax(tail_student_logits, dim=1)
        tail_prob = tail_log_prob.exp()

        tail_terms = torch.where(
            tail_mask,
            tail_prob * (tail_log_prob - student_tail_log_prob),
            torch.zeros_like(tail_prob),
        )

        tail_loss = tail_terms.sum(dim=1).mean() * T * T

        loss = lambda_head * head_loss + lambda_tail * tail_loss

        stats = {
            "head_loss": head_loss.detach(),
            "tail_loss": tail_loss.detach(),
            "boundaries": boundaries.detach(),
            "boundary_mean": boundaries.float().mean().detach(),
        }

        return loss, stats


def sdkd_loss(
        stu_predictions,
        tea_predictions,
        gt_classes,
        ina_rank_weight=0.5,
        temperature=1.0,
        head_weight=1.0,
        tail_weight=0.25,
):
    logits_student, _ = stu_predictions
    logits_teacher, _ = tea_predictions
    target = torch.cat(tuple(gt_classes), 0).reshape(-1)

    loss_ina_rank, _ = INARankLoss()(
        logits_student, logits_teacher, target
    )
    loss_ahtkd, _ = AHTKDLoss()(
        logits_student,
        logits_teacher,
        target,
        temperature,
        head_weight,
        tail_weight,
    )
    return {
        "loss_ina_rank": ina_rank_weight * loss_ina_rank,
        "loss_ahtkd": loss_ahtkd,
    }
