import torch.nn as nn


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(
        a - a.mean(1, keepdim=True),
        b - b.mean(1, keepdim=True),
        eps,
    )


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DISTLoss(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0):
        super(DISTLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits_student, logits_teacher):
        probability_student = logits_student.softmax(dim=1)
        probability_teacher = logits_teacher.softmax(dim=1)
        inter_loss = inter_class_relation(probability_student, probability_teacher)
        intra_loss = intra_class_relation(probability_student, probability_teacher)
        return self.beta * inter_loss + self.gamma * intra_loss
