import torch
import torch.nn.functional as F


def multi_scale_logits(feature_map, classifier, scales=(1, 2, 4)):
    pooled = [
        F.adaptive_avg_pool2d(feature_map, (scale, scale)).flatten(2)
        for scale in scales
    ]
    regions = torch.cat(pooled, dim=2).transpose(1, 2)
    batch_size, region_count, channels = regions.shape
    logits = classifier(regions.reshape(batch_size * region_count, channels))
    return logits.reshape(batch_size, region_count, -1).transpose(1, 2)
