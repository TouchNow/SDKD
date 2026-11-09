from __future__ import division, print_function

import sys
import time

import torch
# from .distiller_zoo import DistillKL2
import torch.nn as nn
import torch.nn.functional as F

from .util import AverageMeter, accuracy, reduce_tensor


def _format_train_run_summary(run_idx, run_times, run_seeds, current_best, previous_bests):
    if run_times <= 1:
        return ""
    previous_bests = list(previous_bests)
    best_values = previous_bests + [current_best]
    best_so_far = max(best_values) if best_values else current_best
    previous_text = ",".join("{:.2f}".format(acc) for acc in previous_bests)
    seeds_text = ",".join(str(seed) for seed in run_seeds)
    return (
        "Run [{}/{}] Seeds [{}] BestSoFar {:.2f} CurrentBest {:.2f} PreviousBests [{}]"
        .format(run_idx, run_times, seeds_text, best_so_far, current_best, previous_text)
    )


def _format_train_distill_summary(
        epoch, idx, n_batch, gpu, losses, loss_cls, loss_kl,
        loss_head, loss_tail, gap_k, top1, top5, run_summary="",
        ina_rank=None, ina_rank_pairs=None,
):
    if isinstance(gap_k, (list, tuple)):
        gap_k_text = "[" + ",".join(str(int(k)) for k in gap_k if int(k) != 1) + "]"
    else:
        gap_k_text = "{:.2f}".format(gap_k.avg)
    run_text = (run_summary + "\t") if run_summary else ""
    relation_text = (
        "INARank {:.2f}\tPairs {:.0f}\t".format(
            ina_rank.avg, ina_rank_pairs.avg
        )
        if ina_rank is not None
        else ""
    )
    return ('Epoch: [{0}][{1}/{2}]\t'
            'GPU {3}\t'
            '{run_text}'
            'Loss {loss.avg:.2f}\t'
            'CE {loss_cls.avg:.2f}\t'
            'KD {loss_kl.avg:.2f}\t'
            'HeadLoss {loss_head.avg:.2f}\t'
            'TailLoss {loss_tail.avg:.2f}\t'
            '{relation_text}'
            'GapK {gap_k}\t'
            'Acc@1 {top1.avg:.2f}\t'
            'Acc@5 {top5.avg:.2f}'.format(
                epoch, idx, n_batch, gpu, loss=losses, loss_cls=loss_cls,
                loss_kl=loss_kl, loss_head=loss_head, loss_tail=loss_tail,
                relation_text=relation_text, gap_k=gap_k_text,
                top1=top1, top5=top5, run_text=run_text
            ))


def _compute_ina_rank_loss(criterion_kd, logit_s, logit_t, target, beta):
    if beta == 0:
        return logit_s.new_zeros(()), 0
    loss_kd, ina_rank_stats = criterion_kd(logit_s, logit_t, target)
    return loss_kd, int(ina_rank_stats['pairs'].item())


def _weighted_loss_components(
        opt, loss_cls, loss_div, loss_kd, loss_gap_ht=None,
):
    if not torch.is_tensor(loss_kd):
        loss_kd = loss_cls.new_tensor(float(loss_kd))
    if loss_gap_ht is None:
        loss_gap_ht = loss_cls.new_zeros(())
    weighted_cls = opt.gamma * loss_cls
    if opt.distill == 'tgeo':
        weighted_cls = loss_cls.new_zeros(())
        weighted_div = loss_cls.new_zeros(())
        weighted_kd = loss_kd
        weighted_gap_ht = loss_cls.new_zeros(())
    elif opt.distill == 'sdkd':
        weighted_div = loss_cls.new_zeros(())
        weighted_kd = opt.beta * loss_kd
        weighted_gap_ht = loss_gap_ht
    elif opt.distill == 'fcka':
        weighted_div = opt.alpha * loss_div
        weighted_kd = opt.fcka_weight * loss_kd
        weighted_gap_ht = loss_cls.new_zeros(())
    elif opt.distill in ['dynamic_dual', 'rekd', 'ldrld']:
        weighted_div = loss_div
        weighted_kd = loss_kd
        weighted_gap_ht = loss_cls.new_zeros(())
    else:
        weighted_div = opt.alpha * loss_div
        weighted_kd = opt.beta * loss_kd
        weighted_gap_ht = loss_cls.new_zeros(())
    return weighted_cls, weighted_div, weighted_kd, weighted_gap_ht


def _gap_head_tail_enabled(opt):
    return (opt.gap_head != 0) or (opt.gap_tail != 0)


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)

    end = time.time()
    for idx, batch_data in enumerate(train_loader):

        input, target = batch_data

        data_time.update(time.time() - end)

        # input = input.float()
        if opt.gpu is not None:
            input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        # output = model(input, is_feat=True)

        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, target, topk=(1, 5))
        top1.update(metrics[0].item(), input.size(0))
        top5.update(metrics[1].item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.avg:.2f}\t'
                  'Acc@1 {top1.avg:.2f}\t'
                  'Acc@5 {top5.avg:.2f}'.format(
                epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg


def train_distill(epoch, train_loader, module_list, mlp_net, cos_value, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()

    if mlp_net is not None:
        mlp_net.train()

    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]
    criterion_gap_ht = criterion_list[3] if len(criterion_list) > 3 else None

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_ce = AverageMeter()
    loss_kl = AverageMeter()
    loss_other_kd = AverageMeter()
    loss_gap_ht_meter = AverageMeter()
    ina_rank = AverageMeter()
    ina_rank_pairs = AverageMeter()
    head_kl = AverageMeter()
    tail_kl = AverageMeter()
    mass_kl = AverageMeter()
    mask_mean = AverageMeter()
    tgeo_alpha = AverageMeter()
    tgeo_alpha_min = AverageMeter()
    tgeo_alpha_max = AverageMeter()
    tgeo_kd = AverageMeter()
    tgeo_ce = AverageMeter()
    ldrld_local = AverageMeter()
    ldrld_weighted_local = AverageMeter()
    ldrld_llki = AverageMeter()
    ldrld_rntk = AverageMeter()
    gap_ht = AverageMeter()
    gap_ht_head = AverageMeter()
    gap_ht_tail = AverageMeter()
    gap_ht_boundary = AverageMeter()
    gap_k_values = []
    head_k = 0
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)
    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target = data
            if opt.distill == 'semckd' and input.shape[0] < opt.batch_size:
                continue

        if opt.gpu is not None:
            input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        if opt.distill == 'sdd':
            feat_s, logit_s, patch_s = model_s(
                input, is_feat=True, sdd_scales=(1, 2, 4)
            )
        else:
            feat_s, logit_s = model_s(input, is_feat=True)

        with torch.no_grad():
            if opt.distill == 'sdd':
                feat_t, logit_t, patch_t = model_t(
                    input, is_feat=True, sdd_scales=(1, 2, 4)
                )
            else:
                feat_t, logit_t = model_t(input, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        if opt.distill == 'dynamic_dual':
            temp = logit_s.new_tensor([opt.kd_T])
            mask, _, _ = mlp_net(logit_t, logit_s, opt.head_ratio, opt.gumbel_tau)
        elif opt.distill in ['tekap', 'fcka']:
            temp = logit_s.new_tensor([opt.kd_T])
        elif opt.have_mlp:
            temp = mlp_net(logit_t, logit_s, cos_value)  # (teacher_output, student_output)
            temp = opt.t_start + opt.t_end * torch.sigmoid(temp)  # 1,20
            temp = temp.cuda()
        elif opt.distill in ['dist', 'nkd', 'sdd', 'rekd', 'tgeo', 'ldrld', 'sdkd']:
            temp = logit_s.new_tensor([opt.kd_T])
        else:
            temp = (opt.kd_T * torch.ones(1)).cuda()

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        if opt.distill == 'sdd':
            loss_div = criterion_kd(patch_s, patch_t, target)
        elif opt.distill == 'tekap':
            loss_div = criterion_kd(logit_s, logit_t, temp)
        elif opt.distill in ['dynamic_dual', 'dist', 'nkd', 'rekd', 'tgeo', 'ldrld', 'sdkd']:
            loss_div = logit_s.new_zeros(())
        else:
            loss_div = criterion_div(logit_s, logit_t, temp)
        loss_gap_ht = logit_s.new_zeros(())

        # other kd beyond KL divergence
        if opt.distill in ['kd', 'sdd', 'tekap']:
            loss_kd = 0
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'fcka':
            loss_kd = criterion_kd(feat_s[-2], feat_t[-2])
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'srrl':
            cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else \
            model_t.get_feat_modules()[-1]
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
        elif opt.distill == 'dkd':
            loss_kd = criterion_kd(logit_s, logit_t, target, opt.dkd_alpha, opt.dkd_beta, temp)
        elif opt.distill == 'dist':
            loss_kd = criterion_kd(logit_s, logit_t)
        elif opt.distill == 'nkd':
            loss_kd = criterion_kd(logit_s, logit_t, target)
        elif opt.distill == 'rekd':
            loss_kd, _ = criterion_kd(
                logit_s, logit_t, opt.rekd_k, temp, opt.rekd_alpha, opt.rekd_beta
            )
        elif opt.distill == 'tgeo':
            loss_kd, kd_stats = criterion_kd(logit_s, logit_t, target, None, temp)
            tgeo_alpha.update(kd_stats['alpha_mean'].item(), input.size(0))
            tgeo_alpha_min.update(kd_stats['alpha_min'].item(), input.size(0))
            tgeo_alpha_max.update(kd_stats['alpha_max'].item(), input.size(0))
            tgeo_kd.update(kd_stats['kd_loss'].item(), input.size(0))
            tgeo_ce.update(kd_stats['ce_loss'].item(), input.size(0))
        elif opt.distill == 'ldrld':
            loss_kd, kd_stats = criterion_kd(
                logit_s, logit_t, opt.ldrld_depth, temp,
                opt.ldrld_alpha, opt.ldrld_beta
            )
            ldrld_local.update(kd_stats['local'].item(), input.size(0))
            ldrld_weighted_local.update(kd_stats['weighted_local'].item(), input.size(0))
            ldrld_llki.update(kd_stats['llki'].item(), input.size(0))
            ldrld_rntk.update(kd_stats['rntk'].item(), input.size(0))
        elif opt.distill == 'sdkd':
            loss_kd, pair_count = _compute_ina_rank_loss(
                criterion_kd, logit_s, logit_t, target, opt.beta
            )
            if pair_count:
                ina_rank.update(
                    (opt.beta * loss_kd).detach().item(), pair_count
                )
            ina_rank_pairs.update(pair_count)
            if _gap_head_tail_enabled(opt):
                loss_gap_ht, gap_stats = criterion_gap_ht(
                    logit_s, logit_t, target, opt.kd_T,
                    opt.gap_head, opt.gap_tail,
                )
                gap_k_values = gap_stats['boundaries'].detach().cpu().tolist()
                gap_ht.update(loss_gap_ht.detach().item(), input.size(0))
                gap_ht_head.update(
                    (opt.gap_head * gap_stats['head_loss']).item(),
                    input.size(0),
                )
                gap_ht_tail.update(
                    (opt.gap_tail * gap_stats['tail_loss']).item(),
                    input.size(0),
                )
                gap_ht_boundary.update(gap_stats['boundary_mean'].item(), input.size(0))
        elif opt.distill == 'dynamic_dual':
            loss_kd, kd_stats = criterion_kd(
                logit_s, logit_t, mask, opt.kd_T,
                opt.dual_head_weight, opt.dual_tail_weight, opt.mass_weight
            )
            head_kl.update(kd_stats['head_kl'].item(), input.size(0))
            tail_kl.update(kd_stats['tail_kl'].item(), input.size(0))
            mass_kl.update(kd_stats['mass_kl'].item(), input.size(0))
            mask_mean.update(mask.detach().mean().item(), input.size(0))
            head_k = int(round(mask.detach().sum(dim=1).mean().item()))
        else:
            raise NotImplementedError(opt.distill)

        weighted_cls, weighted_div, weighted_kd, weighted_gap_ht = _weighted_loss_components(
            opt, loss_cls, loss_div, loss_kd, loss_gap_ht,
        )
        if opt.distill == 'tgeo':
            loss = loss_kd
        else:
            loss = weighted_cls + weighted_div + weighted_kd + weighted_gap_ht

        loss_ce.update(weighted_cls.item(), input.size(0))
        loss_kl.update(weighted_div.item(), input.size(0))
        loss_other_kd.update(weighted_kd.item(), input.size(0))
        loss_gap_ht_meter.update(weighted_gap_ht.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        metrics = accuracy(logit_s, target, topk=(1, 5))
        top1.update(metrics[0].item(), input.size(0))
        top5.update(metrics[1].item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print(_format_train_distill_summary(
                epoch, idx, n_batch, opt.gpu, losses, loss_ce, loss_kl,
                loss_head=gap_ht_head,
                loss_tail=gap_ht_tail,
                gap_k=gap_k_values,
                top1=top1,
                top5=top5,
                ina_rank=ina_rank if opt.distill == 'sdkd' else None,
                ina_rank_pairs=(
                    ina_rank_pairs if opt.distill == 'sdkd' else None
                ),
            ))
            sys.stdout.flush()
            # print(temp)

    train_stats = {'temp': temp.detach()}
    if opt.distill == 'dynamic_dual':
        train_stats.update({
            'head_k': head_k,
            'mask_mean': mask_mean.avg,
            'head_kl': head_kl.avg,
            'tail_kl': tail_kl.avg,
            'mass_kl': mass_kl.avg,
        })
    elif opt.distill == 'tgeo':
        train_stats.update({
            'alpha_mean': tgeo_alpha.avg,
            'alpha_min': tgeo_alpha_min.avg,
            'alpha_max': tgeo_alpha_max.avg,
            'tgeo_kd': tgeo_kd.avg,
            'tgeo_ce': tgeo_ce.avg,
        })
    elif opt.distill == 'ldrld':
        train_stats.update({
            'ldrld_local': ldrld_local.avg,
            'ldrld_weighted_local': ldrld_weighted_local.avg,
            'ldrld_llki': ldrld_llki.avg,
            'ldrld_rntk': ldrld_rntk.avg,
        })
    elif opt.distill == 'sdkd':
        train_stats.update({
            'gap_ht': gap_ht.avg,
            'gap_ht_head': gap_ht_head.avg,
            'gap_ht_tail': gap_ht_tail.avg,
            'gap_ht_boundary': gap_ht_boundary.avg,
            'ina_rank': ina_rank.avg,
            'ina_rank_pairs': ina_rank_pairs.avg,
        })
    return top1.avg, top5.avg, losses.avg, train_stats


def validate(val_loader, model, criterion, opt):
    """validation"""

    # batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader)

    with torch.no_grad():
        # end = time.time()
        for idx, batch_data in enumerate(val_loader):

            input, target = batch_data

            if opt.gpu is not None:
                input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure accuracy and record loss
            metrics = accuracy(output, target, topk=(1, 5))
            top1.update(metrics[0].item(), input.size(0))
            top5.update(metrics[1].item(), input.size(0))

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Loss {loss.avg:.2f}\t'
                      'Acc@1 {top1.avg:.2f}\t'
                      'Acc@5 {top5.avg:.2f}'.format(
                    idx, n_batch, opt.gpu, loss=losses,
                    top1=top1, top5=top5))

    return [top1.avg, top5.avg, losses.avg]
