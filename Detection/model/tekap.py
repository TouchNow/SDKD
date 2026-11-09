import torch
import torch.nn.functional as F


def tekap_loss(
        student_logits,
        teacher_logits,
        temperature,
        num_augmented,
        noise_weight,
        augmented_weight):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if (
            isinstance(num_augmented, bool)
            or not isinstance(num_augmented, int)
            or num_augmented <= 0):
        raise ValueError("num_augmented must be a positive integer")
    if not 0 <= noise_weight <= 1:
        raise ValueError("noise_weight must be between zero and one")
    if augmented_weight < 0:
        raise ValueError("augmented_weight must be non-negative")

    teacher_logits = teacher_logits.detach()
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="batchmean",
    ) * (temperature ** 2)

    teacher_weight = 1 - noise_weight
    for _ in range(num_augmented):
        synthetic_logits = (
            teacher_weight * teacher_logits
            + noise_weight * torch.randn_like(teacher_logits)
        )
        synthetic_prob = F.softmax(synthetic_logits / temperature, dim=1)
        loss = loss + augmented_weight * F.kl_div(
            student_log_prob,
            synthetic_prob,
            reduction="batchmean",
        ) * (temperature ** 2)

    return loss
