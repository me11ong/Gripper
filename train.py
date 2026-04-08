# Copyright (c) OpenMMLab. All rights reserved.
# mmdet 3.x 호환 버전

import os
import argparse
import copy
import os.path as osp
import time
import warnings

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.utils import get_git_hash
from mmdet.utils import setup_cache_size_limit_of_dynamo

import greedyvig_backbone_260313_mmdet3
import CustomMaskRCNN
import CustomCOCODataset
import CustomTransform


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpointpath, resume from it, while if not specify, '
             'try to auto resume from the latest checkpoint in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # mmdet 3.x dynamo 캐시 제한 설정
    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)

    # cfg-options 적용
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir 설정
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # amp 설정
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print('AmpOptimWrapper is already set, skip setting amp')
        else:
            assert optim_wrapper == 'OptimWrapper', \
                '`amp_train` is only supported when the optimizer wrapper type is `OptimWrapper`'
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # auto_scale_lr 설정
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file.')

    # resume 설정
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # validation 여부
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    # Runner로 학습 실행 (mmdet 3.x 방식)
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()