import torch
from torch import nn
import torch.nn.functional as F


class Alignment(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Alignment, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        _, _, h, w = t.shape
        x = self.conv(x)
        return x


class DSKD(nn.Module):
    def __init__(self, cfg, in_channels, out_channels):
        super(DSKD, self).__init__()
        self.cfg = cfg
        # for i in cfg.DSKD.LOC:
        #     setattr(self, 'align' + str(i), Alignment(in_channels[i], out_channels[i]))

    def forward(self, g_s, g_t):
        for i in self.cfg.DSKD.LOC:
            # g_s[i] = getattr(self, 'align' + str(i))(g_s[i], g_t[i]).flatten(1)
            g_s[i] = g_s[i].flatten(1)
            g_t[i] = g_t[i].flatten(1)
        loss_angle = [self.cal_angle_loss(g_s[i], g_t[i]) for i in self.cfg.DSKD.LOC]  # [i]
        loss_angle = sum(loss_angle)
        loss_dist = [self.cal_distance_loss(g_s[i], g_t[i]) for i in self.cfg.DSKD.LOC]  # [i]
        loss_dist = sum(loss_dist)
        return loss_angle, loss_dist

    # def cal_angle_loss(self, feat_s, feat_t):
    #     norm_s = feat_s / feat_s.norm(p=2, dim=-1, keepdim=True)
    #     norm_t = feat_t / feat_t.norm(p=2, dim=-1, keepdim=True)
    #     angle_loss = (1 - (norm_s * norm_t).sum(dim=-1, keepdim=True)).sum() / feat_s.size(0)
    #     return angle_loss

    def cal_angle_loss(self, feat_s, feat_t, eps=1e-8):
        cos_sim = F.cosine_similarity(feat_s, feat_t, dim=-1, eps=eps)  # [B]
        angle_loss = 1.0 - cos_sim.mean()

        return angle_loss

    def cal_distance_loss(self, feat_s, feat_t, T=1.0):  # flatten
        # student
        norm_s = self.norm_distance(feat_s, is_student=True, temperture=T)
        # teacher
        norm_t = self.norm_distance(feat_t, temperture=T)
        dist_loss = nn.KLDivLoss(reduction='batchmean')(norm_s, norm_t)
        return dist_loss

    def norm_distance(self, feat, is_student=False, temperture=1.0):
        feat = self.pdist(feat)
        b, _ = feat.shape
        all_mean = feat.mean(dim=-1, keepdim=True)  # b*1
        nondiag_mean = all_mean * b / (b - 1)  # b*1
        diag = torch.eye(b, device=feat.device) * nondiag_mean  # b*b
        feat_sub_mean = (feat - nondiag_mean) + diag  # onlg consider nondiag mean
        feat_std = (torch.pow(feat_sub_mean, 2).sum(dim=-1, keepdim=True) / (b - 2)).sqrt()  # unbias
        feat = feat_sub_mean / feat_std
        mask = torch.eye(b, device=feat.device) * 1e16
        feat = feat + mask
        if is_student:
            norm_feat = F.softmax(-feat * temperture, dim=-1)
            norm_feat = norm_feat + torch.eye(b, device=norm_feat.device)
            norm_feat = torch.log(norm_feat)
        else:
            norm_feat = F.softmax(-feat * temperture, dim=-1)
        return norm_feat

    def pdist(self, e, eps=1e-5):
        # e_square = (torch.pow(e, 2)).sum(dim=1)
        # prod = e @ e.t()
        # res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod)
        # return res

        res = torch.cdist(e, e, p=2).pow(2)
        res = res.clamp(min=eps)  # 防负值
        return res


def build_dskd_trans(cfg):
    in_channels = [256, 256, 256, 256, 256]
    out_channels = [256, 256, 256, 256, 256]
    model = DSKD(cfg, in_channels, out_channels)
    return model
