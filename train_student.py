"""
the general training framework
"""

from __future__ import print_function

import argparse
import copy
import json
# import tensorboard_logger as tb_logger
import logging
import math
import os
import random
import re
import secrets
import socket
import time

import numpy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from crd.criterion import CRDLoss
from dataset.cifar100 import (get_cifar100_dataloaders,
                              get_cifar100_dataloaders_sample)
from distiller_zoo import AHTKDLoss, DISTLoss, FCKALoss, INARankLoss, NKDLoss, PKT, SDDKDLoss, TeKAPLoss, DistillKL, DKDloss, DynamicDualKDLoss, LDRLDLoss, ReKDLoss, Similarity, TGeoKDLoss, VIDLoss
from helper.loops import train_distill as train
from helper.loops import validate
from helper.util import (adjust_learning_rate, parser_config_save,
                         reduce_tensor, save_dict_to_json)
from models import model_dict

from models.dynamic_mask import DynamicTopKMaskGenerator
from models.temp_global import Global_T
from models.util import SRRL, ConvReg, Embed, LinearEmbed, SelfA
from transfer_student import get_transfer_data_root, run_transfer_evaluations

split_symbol = '~' if os.name == 'nt' else ':'
MAX_SEED = 2 ** 32 - 1


def _get_pretrained_root(hostname=None):
    hostname = socket.gethostname() if hostname is None else hostname
    if hostname.startswith("autodl"):
        return "/root/autodl-tmp/kd/pretrained/"
    if hostname.startswith("optics"):
        return "/home/optics/chendingyao/pretrained/"
    return "/mnt/home/chendingyao/pretrained/pretrained_kd/"


def _teacher_checkpoint(filename):
    return os.path.join(_get_pretrained_root(), filename)


def _resolve_beta(distill, beta):
    if beta is not None:
        return beta
    return 0.5 if distill == 'sdkd' else 0.0


def _validate_sdd_configuration(distill, dataset, model_t, model_s):
    if distill != 'sdd':
        return
    if dataset != 'cifar100':
        raise ValueError('SDD supports CIFAR-100 only')
    supported_models = {'resnet32x4', 'wrn_40_2'}
    if model_t not in supported_models or model_s not in supported_models:
        raise ValueError('SDD supports only resnet32x4 and wrn_40_2')


def _validate_tekap_configuration(distill, dataset):
    if distill == 'tekap' and dataset != 'cifar100':
        raise ValueError('TeKAP supports CIFAR-100 only')


def _validate_transfer_configuration(opt):
    if not opt.transfer_eval:
        return
    expected = {
        'dataset': 'cifar100',
        'distill': 'sdkd',
        'model_s': 'resnet8x4',
        'model_t': 'resnet32x4',
    }
    mismatches = [
        '{}={} (expected {})'.format(name, getattr(opt, name), value)
        for name, value in expected.items()
        if getattr(opt, name) != value
    ]
    if mismatches:
        raise ValueError(
            '--transfer_eval requires the SDKD ResNet32x4/ResNet8x4 '
            'CIFAR-100 experiment: {}'.format(', '.join(mismatches))
        )


def _should_save_best_checkpoint(opt, best_model):
    return best_model and (opt.save_model or opt.transfer_eval)


def _should_save_last_checkpoint(opt, epoch):
    return opt.save_model and epoch == opt.epochs


def _run_transfer_if_enabled(opt):
    if not opt.transfer_eval:
        return
    checkpoint_path = os.path.join(
        opt.save_folder, '{}_best.pth'.format(opt.model_s)
    )
    run_transfer_evaluations(
        checkpoint_path,
        opt.model_s,
        opt.transfer_data_root,
        opt.save_folder,
        num_workers=opt.transfer_num_workers,
    )


def _validate_seed_values(seeds):
    if any(seed < 0 or seed > MAX_SEED for seed in seeds):
        raise ValueError(
            "seed values must be in the range 0 through {}".format(MAX_SEED)
        )
    return seeds


def _resolve_run_seeds(seed_spec, run_times):
    if seed_spec is None:
        return [secrets.randbelow(10001) for _ in range(run_times)]

    parts = str(seed_spec).split(',')
    if any(not part.strip() for part in parts):
        raise ValueError("--seed contains an empty value")
    try:
        seeds = [int(part.strip()) for part in parts]
    except ValueError:
        raise ValueError("--seed must be an integer or comma-separated integers")

    if len(seeds) == 1:
        seeds = [seeds[0] + offset for offset in range(run_times)]
    elif len(seeds) != run_times:
        raise ValueError("--seed must contain exactly {} values".format(run_times))
    return _validate_seed_values(seeds)


def _seed_everything(seed, deterministic=False):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print-freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--run_times', type=int, default=1, help='number of independent training runs')
    parser.add_argument('--seed', type=str, default=None,
                        help='base seed or comma-separated seed for each run')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--experiments_dir', type=str, default='models',
                        help='Directory name to save the model, log, config')
    parser.add_argument('--experiments_name', type=str, default='baseline')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'],
                        help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet20',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'ResNet18', 'ResNet34', 'resnet8x4_double',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'wrn_50_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet18Double'])
    parser.add_argument('--path-t', type=str, default=_teacher_checkpoint('resnet56-(72.34).pth'), help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd',
                        choices=['kd', 'similarity', 'fcka', 'vid', 'pkt', 'crd', 'dkd', 'dist', 'nkd', 'sdd', 'tekap', 'rekd', 'tgeo', 'ldrld',
                                 'sdkd', 'srrl', 'dynamic_dual'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', '--ce_weight', dest='gamma', type=float, default=1.0,
                        help='weight for classification')
    parser.add_argument('-a', '--alpha', '--kd_weight', dest='alpha', type=float, default=1.0,
                        help='weight balance for KD')
    parser.add_argument('-b', '--beta', dest='beta', type=float, default=None,
                        help='weight balance for other distillation losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='default temperature for KD distillation')

    # Feature CKA distillation
    parser.add_argument('--fcka_weight', default=5.0, type=float)

    # Scale Decoupled Distillation
    parser.add_argument('--sdd_beta', default=2.0, type=float)

    # Teacher Knowledge Augmentation
    parser.add_argument('--tekap_num_teachers', default=3, type=int)
    parser.add_argument('--tekap_noise_weight', default=0.1, type=float)
    parser.add_argument('--tekap_teacher_weight', default=0.9, type=float)
    parser.add_argument('--tekap_aug_weight', default=0.8, type=float)

    # Dynamic dual-channel distillation
    parser.add_argument('--head_ratio', type=float, default=0.3)
    parser.add_argument('--gumbel_tau', type=float, default=1.0)
    parser.add_argument('--mask_hidden_dim', type=int, default=128)
    parser.add_argument('--dual_head_weight', type=float, default=1.0)
    parser.add_argument('--dual_tail_weight', type=float, default=1.0)
    parser.add_argument('--mass_weight', type=float, default=1.0)

    # CTKD distillation
    parser.add_argument('--have_mlp', type=int, default=0)
    parser.add_argument('--mlp_name', type=str, default='global')
    parser.add_argument('--t_start', type=float, default=1)
    parser.add_argument('--t_end', type=float, default=20)
    parser.add_argument('--cosine_decay', type=int, default=1)
    parser.add_argument('--decay_max', type=float, default=10)
    parser.add_argument('--decay_min', type=float, default=0)
    parser.add_argument('--decay_loops', type=float, default=10)

    # DKD distillation
    parser.add_argument('--dkd_alpha', default=1, type=float)
    parser.add_argument('--dkd_beta', default=8, type=float)

    # DIST distillation
    parser.add_argument('--dist_beta', default=1.0, type=float)
    parser.add_argument('--dist_gamma', default=1.0, type=float)

    # NKD distillation
    parser.add_argument('--nkd_temp', default=3.0, type=float)
    parser.add_argument('--nkd_gamma', default=1.5, type=float)

    # ReKD distillation
    parser.add_argument('--rekd_k', default=3, type=int)
    parser.add_argument('--rekd_alpha', default=1.0, type=float)
    parser.add_argument('--rekd_beta', default=1.0, type=float)

    # TGeo-KD distillation
    parser.add_argument('--tgeo_hidden_dim', default=128, type=int)

    # LDRLD distillation
    parser.add_argument('--ldrld_depth', default=7, type=int)
    parser.add_argument('--ldrld_alpha', default=7.0, type=float)
    parser.add_argument('--ldrld_beta', default=4.0, type=float)
    parser.add_argument('--ldrld_irw_eps', default=1.5, type=float)
    parser.add_argument('--ldrld_erd_delta', default=2.0, type=float)
    parser.add_argument('--ldrld_erd_lambda', default=0.05, type=float)

    # SDKD distillation
    parser.add_argument('--gap_head', default=1.0, type=float)
    parser.add_argument('--gap_tail', default=6.0, type=float)

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--transfer_eval', action='store_true')
    parser.add_argument(
        '--transfer_data_root', type=str, default=get_transfer_data_root()
    )
    parser.add_argument('--transfer_num_workers', type=int, default=8)

    # switch for edge transformation
    parser.add_argument('--no_edge_transform', action='store_true')  # default=false

    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:8080', type=str,
                        help='url used to set up distributed training')

    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')

    opt = parser.parse_args()
    if opt.run_times < 1:
        raise ValueError('--run_times must be >= 1')

    opt.beta = _resolve_beta(opt.distill, opt.beta)

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard
    opt.model_path = './save/student_model'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)
    _validate_transfer_configuration(opt)
    _validate_sdd_configuration(
        opt.distill, opt.dataset, opt.model_t, opt.model_s
    )
    _validate_tekap_configuration(opt.distill, opt.dataset)

    opt.model_name = os.path.join(opt.experiments_dir, opt.experiments_name)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    parser_config_save(opt, opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    filename = os.path.basename(model_path)
    saved_teacher_match = re.match(r'(.+)_(?:best|last)\.pth$', filename)
    if saved_teacher_match:
        return saved_teacher_match[1]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    if segments[0] == 'resnext50':
        return segments[0] + '_' + segments[1]
    filename_match = re.match(r'([A-Za-z0-9_]+)-', filename)
    if filename_match:
        return filename_match[1]
    return segments[0]


def _format_epoch_eval_summary(epoch, test_acc, test_acc_top5, best_acc):
    return (
        "Epoch {} test: Acc@1 {:.2f} Acc@5 {:.2f} Best Acc@1 {:.2f}"
        .format(epoch, test_acc, test_acc_top5, best_acc)
    )


def _format_run_start_summary(run_idx, run_times, seed):
    return "Run [{}/{}] Seed {} start".format(run_idx, run_times, seed)


def _format_distillation_method(distill):
    return "Distillation method: {}".format(distill)


def _should_print_distillation_method(multiprocessing_distributed, rank):
    return not multiprocessing_distributed or rank == 0


def _format_run_status_summary(run_idx, run_times, run_seeds, best_so_far, run_best_accs):
    previous = run_best_accs[:-1]
    previous_text = ",".join("{:.2f}".format(acc) for acc in previous)
    seeds_text = ",".join(str(seed) for seed in run_seeds)
    current_best = run_best_accs[-1]
    return (
        "Run [{}/{}] Seeds [{}] BestSoFar {:.2f} CurrentBest {:.2f} PreviousBests [{}]"
        .format(run_idx, run_times, seeds_text, best_so_far, current_best, previous_text)
    )


def _format_all_run_summary(run_best_accs):
    run_best_text = ",".join("{:.2f}".format(acc) for acc in run_best_accs)
    return "Best@1 [{}] | Global {:.2f}".format(
        run_best_text, max(run_best_accs)
    )


def _format_validation_run_summary(previous_run_best_accs, current_best):
    return "{} | Current {:.2f}".format(
        _format_all_run_summary(list(previous_run_best_accs) + [current_best]),
        current_best,
    )


def _prepare_run_option(opt, run_idx, run_seed):
    run_opt = copy.copy(opt)
    run_opt.seed = run_seed
    if opt.run_times > 1:
        run_opt.model_name = os.path.join(opt.model_name, 'run-{}'.format(run_idx))
        run_opt.save_folder = os.path.join(opt.model_path, run_opt.model_name)
        if not os.path.isdir(run_opt.save_folder):
            os.makedirs(run_opt.save_folder)
    parser_config_save(run_opt, run_opt.save_folder)
    return run_opt


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    # print(model_t)
    # assert 1 == 0
    model = model_dict[model_t](num_classes=n_cls)
    # TODO: reduce size of the teacher saved in train_teacher.py
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}

    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])

    print('==> done')
    return model


class CosineDecay(object):
    def __init__(self, max_value, min_value, num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class LinearDecay(object):
    def __init__(self, max_value, min_value, num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops - 1
        value = (self._max_value - self._min_value) / self._num_loops
        value = i * (-value)

        return value


def compute_teacher_class_mean(train_loader, model_t, n_cls, opt):
    model_t.eval()
    device = next(model_t.parameters()).device
    class_sum = torch.zeros(n_cls, n_cls, device=device)
    class_count = torch.zeros(n_cls, 1, device=device)

    with torch.no_grad():
        for data in train_loader:
            input, target = data[0], data[1]
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            _, logit_t = model_t(input, is_feat=True)
            prob_t = torch.softmax(logit_t, dim=1)
            class_sum.index_add_(0, target, prob_t)
            class_count.index_add_(0, target, torch.ones(target.size(0), 1, device=device))

    if opt.multiprocessing_distributed:
        dist.all_reduce(class_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_count, op=dist.ReduceOp.SUM)

    class_mean = class_sum / class_count.clamp_min(1.0)
    missing_class = class_count.squeeze(1) == 0
    if missing_class.any():
        class_mean[missing_class] = 1.0 / n_cls
    return class_mean.detach()


total_time = time.time()
best_acc = 0


def main():
    opt = parse_option()
    run_seeds = _resolve_run_seeds(opt.seed, opt.run_times)
    opt.run_seeds = list(run_seeds)

    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        if opt.run_times > 1:
            raise NotImplementedError('--run_times > 1 is not supported with --multiprocessing-distributed')
        run_opt = _prepare_run_option(opt, 1, run_seeds[0])
        run_opt.run_idx = 1
        run_opt.previous_run_best_accs = []
        run_opt.current_run_best_acc = 0.0
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        run_opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print(_format_run_start_summary(1, run_opt.run_times, run_opt.seed))
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, run_opt))
        _run_transfer_if_enabled(run_opt)
    else:
        run_best_accs = []
        for run_idx, run_seed in enumerate(run_seeds, start=1):
            run_opt = _prepare_run_option(opt, run_idx, run_seed)
            run_opt.run_idx = run_idx
            run_opt.previous_run_best_accs = list(run_best_accs)
            run_opt.current_run_best_acc = 0.0
            print(_format_run_start_summary(run_idx, opt.run_times, run_opt.seed))
            run_best = main_worker(None if ngpus_per_node > 1 else run_opt.gpu_id, ngpus_per_node, run_opt)
            _run_transfer_if_enabled(run_opt)
            run_best_accs.append(run_best)
        print(_format_all_run_summary(run_best_accs))


def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if _should_print_distillation_method(
            opt.multiprocessing_distributed, getattr(opt, 'rank', None)):
        print(_format_distillation_method(opt.distill))

    _seed_everything(opt.seed, deterministic=opt.deterministic)

    if opt.dataset != 'cifar100':
        raise NotImplementedError(opt.dataset)
    n_cls = 100

    # model
    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)

    module_args = {'num_classes': n_cls}
    model_s = model_dict[opt.model_s](**module_args)

    data = torch.randn(2, 3, 32, 32)

    mlp = None

    if opt.distill == 'dynamic_dual':
        mlp = DynamicTopKMaskGenerator(n_cls, opt.mask_hidden_dim)
    elif opt.distill == 'fcka':
        pass
    elif opt.have_mlp and opt.distill != 'tekap':
        if opt.mlp_name == 'global':
            mlp = Global_T()
        else:
            print('mlp name wrong')

    model_t.eval()
    model_s.eval()

    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    if mlp is not None:
        trainable_list.append(mlp)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL()
    if opt.distill == 'kd':
        criterion_kd = DistillKL()
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = 50000
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'fcka':
        criterion_kd = FCKALoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'dkd':
        criterion_kd = DKDloss()
    elif opt.distill == 'dist':
        criterion_kd = DISTLoss(opt.dist_beta, opt.dist_gamma)
    elif opt.distill == 'nkd':
        criterion_kd = NKDLoss(opt.nkd_temp, opt.nkd_gamma)
    elif opt.distill == 'sdd':
        criterion_kd = SDDKDLoss(opt.kd_T, opt.sdd_beta)
    elif opt.distill == 'tekap':
        criterion_kd = TeKAPLoss(
            opt.tekap_num_teachers, opt.tekap_noise_weight,
            opt.tekap_teacher_weight, opt.tekap_aug_weight
        )
    elif opt.distill == 'rekd':
        criterion_kd = ReKDLoss()
    elif opt.distill == 'tgeo':
        criterion_kd = TGeoKDLoss(n_cls, opt.tgeo_hidden_dim)
        trainable_list.append(criterion_kd)
    elif opt.distill == 'ldrld':
        criterion_kd = LDRLDLoss(opt.ldrld_irw_eps, opt.ldrld_erd_delta, opt.ldrld_erd_lambda)
    elif opt.distill == 'sdkd':
        criterion_kd = INARankLoss()
        criterion_gap_ht = AHTKDLoss()
    elif opt.distill == 'dynamic_dual':
        criterion_kd = DynamicDualKDLoss()
    elif opt.distill == 'srrl':
        s_n = feat_s[-1].shape[1]
        t_n = feat_t[-1].shape[1]
        model_fmsr = SRRL(s_n=s_n, t_n=t_n)
        criterion_kd = nn.MSELoss()
        module_list.append(model_fmsr)
        trainable_list.append(model_fmsr)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss
    if opt.distill == 'sdkd':
        criterion_list.append(criterion_gap_ht)

    module_list.append(model_t)

    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list.cuda(opt.gpu)
                distributed_modules = []
                for module in module_list:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                module_list = distributed_modules
                criterion_list.cuda(opt.gpu)
                if mlp is not None:
                    mlp.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list.cuda()
            if mlp is not None:
                mlp.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)

    if opt.distill == 'tgeo':
        print('==> computing teacher class mean for TGeo-KD')
        teacher_class_mean = compute_teacher_class_mean(train_loader, model_t, n_cls, opt)
        criterion_kd.set_teacher_class_mean(teacher_class_mean)

    if opt.cosine_decay:
        gradient_decay = CosineDecay(max_value=opt.decay_max, min_value=opt.decay_min, num_loops=opt.decay_loops)
    else:
        gradient_decay = LinearDecay(max_value=opt.decay_max, min_value=opt.decay_min, num_loops=opt.decay_loops)

    decay_value = 1
    best_acc = 0
    return_pack = validate(val_loader, model_t, criterion_cls, opt)
    test_acc, test_acc_top5, _ = return_pack
    print('teacher acc:', test_acc)
    for epoch in range(1, opt.epochs + 1):

        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        if opt.have_mlp and opt.distill not in ['dynamic_dual', 'fcka']:
            decay_value = gradient_decay.get_value(epoch)

        opt.current_run_best_acc = best_acc
        train_acc, train_acc_top5, train_loss, train_stats = train(epoch, train_loader, module_list, mlp, decay_value,
                                                                   criterion_list, optimizer, opt)

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        return_pack = validate(val_loader, model_s, criterion_cls, opt)
        test_acc, test_acc_top5, _ = return_pack

        best_model = False
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = True
            print(_format_epoch_eval_summary(epoch, test_acc, test_acc_top5, best_acc))
            current_run_best_accs = list(
                getattr(opt, 'previous_run_best_accs', [])
            ) + [best_acc]
            print(_format_run_status_summary(
                opt.run_idx, opt.run_times, opt.run_seeds,
                max(current_run_best_accs), current_run_best_accs,
            ))

            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))

            test_merics = {
                'test_acc': test_acc,
                'test_acc_top5': test_acc_top5,
                'best_acc': best_acc,
                'epoch': epoch,
                'temp': json.dumps(train_stats['temp'].cpu().detach().numpy()[0].tolist()),
                'decay_value': decay_value}
            test_merics.update({k: v for k, v in train_stats.items() if k != 'temp'})

            save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))

            if _should_save_best_checkpoint(opt, best_model):
                torch.save(state, save_file)
            if _should_save_last_checkpoint(opt, epoch):
                last_save_file = os.path.join(
                    opt.save_folder, '{}_last.pth'.format(opt.model_s)
                )
                torch.save(state, last_save_file)
    return best_acc


if __name__ == '__main__':
    main()
