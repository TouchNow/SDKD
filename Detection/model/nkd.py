import torch
import torch.nn.functional as F


def nkd_loss(student_logits, teacher_logits, targets, temperature, gamma):
    teacher_logits = teacher_logits.detach()
    if targets.ndim > 1:
        labels = targets.argmax(dim=1, keepdim=True)
    else:
        labels = targets.reshape(-1, 1)

    student_log_probability = F.log_softmax(student_logits, dim=1)
    teacher_probability = F.softmax(teacher_logits, dim=1)
    target_loss = -(
        teacher_probability.gather(1, labels)
        * student_log_probability.gather(1, labels)
    ).mean()

    non_target_mask = torch.ones_like(student_logits, dtype=torch.bool)
    non_target_mask.scatter_(1, labels, False)
    batch_size = student_logits.shape[0]
    student_non_target = student_logits[non_target_mask].reshape(batch_size, -1)
    teacher_non_target = teacher_logits[non_target_mask].reshape(batch_size, -1)

    student_non_target_log_probability = F.log_softmax(
        student_non_target / temperature, dim=1)
    teacher_non_target_probability = F.softmax(
        teacher_non_target / temperature, dim=1)
    non_target_loss = -(
        teacher_non_target_probability * student_non_target_log_probability
    ).sum(dim=1).mean()

    return target_loss + gamma * temperature ** 2 * non_target_loss
