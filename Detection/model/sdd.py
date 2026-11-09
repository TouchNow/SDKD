import torch
import torch.nn.functional as F


def num_sdd_regions(scales):
    return sum(scale * scale for scale in scales)


def build_sdd_boxes(boxes, scales):
    """Split proposal boxes into region-major multi-scale grids."""
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    widths = x2 - x1
    heights = y2 - y1
    regions = []

    for scale in scales:
        for row in range(scale):
            for column in range(scale):
                regions.append(torch.stack([
                    x1 + widths * column / scale,
                    y1 + heights * row / scale,
                    x1 + widths * (column + 1) / scale,
                    y1 + heights * (row + 1) / scale,
                ], dim=1))

    return torch.cat(regions, dim=0)


def reshape_sdd_logits(logits, proposal_counts, num_regions):
    """Convert image-major ROI logits to [proposal, class, region]."""
    class_count = logits.shape[1]
    split_sizes = [count * num_regions for count in proposal_counts]
    image_logits = logits.split(split_sizes, dim=0)
    reshaped = [
        values.reshape(num_regions, count, class_count).permute(1, 2, 0)
        for values, count in zip(image_logits, proposal_counts)
        if count > 0
    ]
    if not reshaped:
        return logits.new_empty((0, class_count, num_regions))
    return torch.cat(reshaped, dim=0)


def sdd_loss(student_logits, teacher_logits, targets, temperature, beta):
    """Scale-decoupled KD with official complementary-term weighting."""
    class_count = student_logits.shape[1]
    student_flat = student_logits.permute(2, 0, 1).reshape(-1, class_count)
    teacher_flat = teacher_logits.permute(2, 0, 1).reshape(-1, class_count)

    per_region_loss = F.kl_div(
        F.log_softmax(student_flat / temperature, dim=1),
        F.softmax(teacher_flat / temperature, dim=1),
        reduction="none",
    ).sum(dim=1) * temperature ** 2

    teacher_predictions = teacher_logits.argmax(dim=1)
    global_correct = teacher_predictions[:, 0].eq(targets)
    local_correct = teacher_predictions.eq(targets[:, None])
    complementary = global_correct[:, None] ^ local_correct
    weights = torch.where(
        complementary,
        per_region_loss.new_tensor(beta),
        per_region_loss.new_tensor(1.0),
    ).transpose(0, 1).reshape(-1)

    return (per_region_loss * weights).mean()
