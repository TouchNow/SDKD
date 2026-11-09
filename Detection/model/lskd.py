import torch.nn.functional as F


def standardize_logits(logits):
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    return (logits - mean) / (std + 1e-7)


def lskd_loss(student_logits, teacher_logits, temperature):
    student_logits = standardize_logits(student_logits)
    teacher_logits = standardize_logits(teacher_logits)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(
        log_student,
        teacher_prob,
        reduction="none",
    ).sum(1).mean()
    return loss * temperature ** 2
