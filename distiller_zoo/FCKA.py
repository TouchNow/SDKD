import torch
import torch.nn as nn


class FCKALoss(nn.Module):
    """Feature CKA loss following the RCKA authors' uncentered formula."""

    def __init__(self, eps=1e-12):
        super(FCKALoss, self).__init__()
        self.eps = eps

    def forward(self, feature_s, feature_t):
        feature_s = feature_s.reshape(feature_s.shape[0], -1)
        feature_t = feature_t.reshape(feature_t.shape[0], -1)

        gram_s = torch.mm(feature_s, feature_s.t())
        gram_t = torch.mm(feature_t, feature_t.t())
        if gram_s.dtype in (torch.float16, torch.bfloat16):
            gram_s = gram_s.float()
            gram_t = gram_t.float()
        numerator = (gram_s * gram_t).sum()
        denominator = (
            torch.linalg.vector_norm(gram_s)
            * torch.linalg.vector_norm(gram_t)
        )
        minimum = max(self.eps, torch.finfo(denominator.dtype).tiny)
        denominator = denominator.clamp_min(minimum)
        return -(numerator / denominator)
