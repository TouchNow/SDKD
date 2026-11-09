import torch
import torch.nn as nn
import torch.nn.functional as F


class TGeoFusionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(TGeoFusionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.net(features).squeeze(1)


class TGeoKDLoss(nn.Module):
    def __init__(self, num_classes, hidden_dim=128, fusion_net=None, eps=1e-8):
        super(TGeoKDLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        input_dim = 9 * num_classes
        self.fusion_net = fusion_net if fusion_net is not None else TGeoFusionNet(input_dim, hidden_dim)
        self.register_buffer("teacher_class_mean", torch.eye(num_classes))

    def set_teacher_class_mean(self, teacher_class_mean):
        if teacher_class_mean.shape != self.teacher_class_mean.shape:
            raise ValueError("teacher_class_mean must have shape [num_classes, num_classes]")
        self.teacher_class_mean.copy_(teacher_class_mean.detach().to(self.teacher_class_mean.device))

    def geometry_features(self, logits_student, logits_teacher, target, teacher_class_mean=None):
        student_prob = F.softmax(logits_student, dim=1)
        teacher_prob = F.softmax(logits_teacher.detach(), dim=1)
        ground_truth = F.one_hot(target, num_classes=self.num_classes).to(
            device=logits_student.device,
            dtype=logits_student.dtype,
        )

        class_mean = self.teacher_class_mean if teacher_class_mean is None else teacher_class_mean
        class_mean = class_mean.to(device=logits_student.device, dtype=logits_student.dtype).detach()
        teacher_mean = class_mean[target]

        student_ground = ground_truth - student_prob
        teacher_ground = ground_truth - teacher_prob
        student_teacher = teacher_prob - student_prob
        mean_teacher_ground = ground_truth - teacher_mean
        student_mean_teacher = teacher_mean - student_prob

        return torch.cat(
            [
                student_ground,
                teacher_ground,
                student_teacher,
                mean_teacher_ground,
                student_mean_teacher,
                student_prob,
                teacher_prob,
                teacher_mean,
                ground_truth,
            ],
            dim=1,
        )

    def forward(self, logits_student, logits_teacher, target, teacher_class_mean=None, temperature=4.0):
        temp = self._temperature(logits_student, temperature)
        features = self.geometry_features(logits_student, logits_teacher, target, teacher_class_mean)
        alpha = self.fusion_net(features).clamp(self.eps, 1.0 - self.eps)

        kd_per_sample = (
            F.kl_div(
                F.log_softmax(logits_student / temp, dim=1),
                F.softmax(logits_teacher.detach() / temp, dim=1),
                reduction="none",
            ).sum(dim=1)
            * (temp ** 2)
        )
        ce_per_sample = F.cross_entropy(logits_student, target, reduction="none")
        loss = (alpha * kd_per_sample + (1.0 - alpha) * ce_per_sample).mean()

        stats = {
            "alpha_mean": alpha.detach().mean(),
            "alpha_min": alpha.detach().min(),
            "alpha_max": alpha.detach().max(),
            "kd_loss": kd_per_sample.detach().mean(),
            "ce_loss": ce_per_sample.detach().mean(),
        }
        return loss, stats

    def _temperature(self, logits_student, temperature):
        if torch.is_tensor(temperature):
            return temperature.to(device=logits_student.device, dtype=logits_student.dtype)
        return logits_student.new_tensor(float(temperature))
