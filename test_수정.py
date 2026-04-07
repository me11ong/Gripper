# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, #rfnext_init_model,
                         setup_multi_processes, update_data_root)


import greedyvig_backbone_260313
import CustomMaskRCNN
import CustomCOCODataset as CustomCOCODataset
import CustomTransform

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
import cv2
import json
import numpy as np

from sklearn.metrics import confusion_matrix
from pycocotools import mask as maskUtils

PALETTE = [
    (163, 6, 70), (57, 188, 173), (228, 22, 108), (7, 55, 129), 
    (6, 50, 166), (139, 56, 150), (207, 1, 206), (178, 87, 39), 
    (245, 86, 23), (24, 216, 154), (206, 186, 137), (249, 96, 141), 
    (212, 158, 220), (147, 180, 11), (58, 74, 20), (59, 25, 71), 
    (162, 93, 94), (53, 68, 239), (165, 155, 43), (186, 41, 97), 
    (253, 163, 142), (175, 215, 198), (58, 8, 80), (68, 54, 241), 
    (224, 80, 167), (101, 234, 117), (67, 63, 143), (67, 149, 229), 
    (102, 56, 255), (130, 23, 12), (28, 160, 202), (108, 16, 97), 
    (255, 135, 248), (220, 2, 184), (174, 137, 68), (164, 28, 111), 
    (116, 244, 224), (67, 128, 45), (233, 222, 76), (163, 155, 39), 
    (195, 138, 199), (135, 0, 82), (4, 237, 224), (212, 78, 14), 
    (224, 242, 21), (124, 17, 194), (196, 32, 121), (140, 67, 223), 
    (108, 54, 138), (186, 51, 79), (254, 166, 112), (132, 30, 57), 
    (86, 150, 58), (56, 18, 161), (58, 231, 220), (18, 60, 171), 
    (54, 33, 239), (146, 121, 200), (206, 48, 24), (110, 108, 119), 
    (186, 172, 251), (25, 103, 86), (220, 63, 48), (114, 108, 71), 
    (63, 236, 113), (220, 140, 12), (255, 214, 248), (237, 217, 42), 
    (124, 54, 102), (15, 97, 252), (67, 200, 116), (108, 245, 254), 
    (142, 183, 39), (75, 247, 148), (138, 191, 14), (149, 128, 218), 
    (40, 245, 20), (47, 152, 172), (60, 30, 227), (63, 152, 158), 
    (107, 149, 133), (239, 52, 183), (61, 101, 171), (76, 80, 192), 
    (18, 117, 255), (255, 18, 54), (67, 238, 225), (225, 94, 40), 
    (213, 180, 156), (252, 167, 2), (209, 76, 169), (240, 34, 29), 
    (27, 141, 69), (154, 183, 52), (162, 67, 125), (231, 216, 23), 
    (108, 70, 0), (197, 163, 67), (189, 141, 109), (2, 19, 226), 
    (231, 139, 213), (149, 37, 32), (78, 230, 203), (220, 230, 53), 
    (63, 26, 199), (226, 104, 158), (39, 238, 221), (249, 207, 225), 
    (6, 188, 85), (238, 205, 221), (207, 68, 201), (27, 223, 219), 
    (56, 209, 117), (78, 203, 58), (6, 49, 84), (221, 247, 71), 
    (164, 102, 251), (137, 240, 29), (248, 45, 246), (67, 27, 111), 
    (186, 80, 155), (130, 98, 147), (65, 181, 0), (236, 137, 184), 
    (189, 171, 93), (17, 170, 84), (80, 217, 184), (76, 79, 104), 
    (103, 75, 32), (107, 240, 173), (231, 157, 77), (140, 0, 73), 
    (110, 148, 167), (119, 113, 54), (121, 230, 203), (43, 21, 131), 
    (162, 85, 209), (192, 172, 57), (50, 6, 62), (121, 217, 18), 
    (106, 161, 49), (178, 126, 62), (167, 1, 192), (197, 27, 108), 
    (45, 245, 132), (12, 63, 217), (116, 205, 170), (252, 152, 243), 
    (228, 156, 184), (129, 212, 140), (229, 190, 121), (66, 63, 163), 
    (196, 133, 160), (70, 19, 73), (69, 81, 138), (35, 59, 177), 
    (180, 16, 104), (138, 106, 52), (107, 231, 149), (178, 219, 195), 
    (97, 1, 90), (192, 218, 244), (107, 191, 139), (154, 56, 56), 
    (111, 7, 86)]

def compute_pr_metrics(coco_gt, coco_dt, iou_thrs=[0.5, 0.7, 0.9], iouType = 'bbox'):
    cocoEval = COCOeval(coco_gt, coco_dt, iouType)
    
    # IoU threshold 설정
    cocoEval.params.iouThrs = np.array(iou_thrs)
    
    cocoEval.evaluate()
    cocoEval.accumulate()

    precision = cocoEval.eval['precision']
    # shape: [T, R, K, A, M]

    results = {}

    for i, thr in enumerate(iou_thrs):
        pr = precision[i, :, :, 0, -1]  # area=all, maxDets=max
        pr = pr[pr > -1]  # valid 값만

        results[f'PR@{int(thr*100)}'] = np.mean(pr)

    return results

def draw_bboxes_with_labels(img, result, cls_name, score_thr=0.5):
    bbox_results = result[0][0]
    mask_results = result[0][1]

    for cls_id, bboxes in enumerate(bbox_results):

        if len(bboxes) == 0:
            continue

        masks = mask_results[cls_id]

        for bbox, mask in zip(bboxes, masks):
            x1, y1, x2, y2, score = bbox
            if score < score_thr:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            if mask is not None:
                mask = mask.astype(bool)

                color = PALETTE[cls_id]
                alpha = 0.6

                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[mask] = color

                img = cv2.addWeighted(img, 1.0, colored_mask, alpha, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2),
                          color=(0, 255, 0), thickness=2)

            text = f'{cls_id}:{cls_name}'
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                img,
                (x1, y1 - th - 4),
                (x1 + tw, y1),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                img,
                text,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    return img
def compute_metrics(all_predictions, all_ground_truths, num_classes):
    """
    confusion matrix 기반 mACC, mIoU, fwIoU, pACC 계산
    
    Args:
        all_predictions (list of dict): [{'image_id': int, 'class': int, 'mask': ndarray(bool)}, ...]
        all_ground_truths (list of dict): [{'image_id': int, 'class': int, 'mask': ndarray(uint8)}, ...]
        num_classes (int): 전체 클래스 수 (배경 포함)
    """
    gt_dict = {}
    for g in all_ground_truths:
        if g['image_id'] not in gt_dict:
            gt_dict[g['image_id']] = []
        gt_dict[g['image_id']].append(g)
    
    pred_dict = {}
    for p in all_predictions:
        if p['image_id'] not in pred_dict:
            pred_dict[p['image_id']] = []
        pred_dict[p['image_id']].append(p)

    y_true_list = []
    y_pred_list = []

    for image_id in gt_dict.keys():
        gt_items = gt_dict[image_id]
        pred_items = pred_dict.get(image_id, [])

        if not gt_items:
            continue
        h, w = gt_items[0]['mask'].shape
        gt_class_mask = np.zeros((h, w), dtype=int)
        pred_class_mask = np.zeros((h, w), dtype=int)

        for gt in gt_items:
            gt_class_mask[gt['mask'] > 0] = gt['class']

        for pred in pred_items:
            pred_class_mask[pred['mask'] > 0] = pred['class']

        y_true_list.append(gt_class_mask.flatten())
        y_pred_list.append(pred_class_mask.flatten())

    if not y_true_list:
        raise ValueError("No matching predictions and ground truths found.")

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    acc = TP / (TP + FN + 1e-10)
    mACC = np.nanmean(acc)

    IoU = TP / (TP + FP + FN + 1e-10)
    mIoU = np.nanmean(IoU)

    freq = np.sum(cm, axis=1) / np.sum(cm)
    fwIoU = (freq * IoU).sum()

    pACC = TP.sum() / cm.sum()

    Dice = (2 * TP) / (2 * TP + FP + FN + 1e-10)   # 클래스별 Dice
    mDice = np.nanmean(Dice)  

    return {
        "mACC": mACC,
        "mIoU": mIoU,
        "fwIoU": fwIoU,
        "pACC": pACC,
        'mDice':mDice
    }
# def single_gpu_test(model,
#                     data_loader,
#                     work_dir,
#                     show=False,
#                     out_dir=None,
#                     show_score_thr=0.5,
#                     ):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset

#     prog_bar = mmcv.ProgressBar(len(dataset))
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, **data)
#             img_metas = data['img_metas'][0].data[0]
#             img_meta = img_metas[0]
            
#             #img_id = dataset.data_infos[i]['id'] 
            
#             #dump_path = os.path.join(dump_dir, 'preds.jsonl')

#             # def tolist_safe(x):
#             #     if isinstance(x, np.ndarray):
#             #         return x.tolist()
#             #     if hasattr(x, 'detach'):
#             #         return x.detach().cpu().tolist()
#             #     return x

#             #dump_result = []

#             # bbox 결과만 우선 dump (mask는 나중에)
#             bbox_results = result[0][0]  # list[num_classes]
#             mask_results = result[0][1]
            
#             # for cls_id, bboxes in enumerate(bbox_results):
 
#             #     if len(bboxes) == 0:
#             #         continue
#             #     for bbox in bboxes:
#             #         if len(bbox)!=0:  
#             #             dump_result.append({
#             #                 'img_id': img_id,
#             #                 'label': int(cls_id),
#             #                 'bbox': tolist_safe(bbox[:4]),
#             #                 "mask" : mask_results[cls_id][0].astype(np.uint8).tolist(),
#             #                 'score': float(bbox[4])
#             #             })

#             # with open(dump_path, 'a') as f:
#             #     json.dump(dump_result, f, indent=4)
#             #     f.write('\n')

#         batch_size = len(result)
#         if show or out_dir:
            
#             dump_dir = os.path.join(work_dir, '시각화')
#             os.makedirs(dump_dir, exist_ok=True)

#             if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
#                 img_tensor = data['img'][0]
#             else:
#                 img_tensor = data['img'][0].data[0]

#             img_metas = data['img_metas'][0].data[0]
#             imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
#             assert len(imgs) == len(img_metas)
            
#             for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
#                 h, w, _ = img_meta['img_shape']
#                 img_show = img[:h, :w, :]

#                 ori_h, ori_w = img_meta['ori_shape'][:-1]
#                 img_show = mmcv.imresize(img_show, (ori_w, ori_h))

#                 if osp.exists(osp.join(out_dir, img_meta['ori_filename'])):
#                     img_show = cv2.imread(osp.join(out_dir, img_meta['ori_filename']))

#                 result_img = draw_bboxes_with_labels(
#                     img_show,
#                     result,
#                     img_metas[0]['texts'].data[0],
#                     score_thr=show_score_thr
#                 )

#                 out_file = os.path.join(dump_dir, f'{img_meta["ori_filename"]}')
#                 cv2.imwrite(out_file, result_img)

#         # encode mask results
#         if isinstance(result[0], tuple):
#             result = [(bbox_results, encode_mask_results(mask_results))
#                       for bbox_results, mask_results in result]
#         # This logic is only used in panoptic segmentation test.
#         elif isinstance(result[0], dict) and 'ins_results' in result[0]:
#             for j in range(len(result)):
#                 bbox_results, mask_results = result[j]['ins_results']
#                 result[j]['ins_results'] = (bbox_results,
#                                             encode_mask_results(mask_results))

#         results.extend(result)

#         for _ in range(batch_size):
#             prog_bar.update()
    

#     return results

def polygons_to_mask(polygons, height, width):
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)
    return mask.astype(np.uint8)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.5,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
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
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)


    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # init rfnext if 'RFSearchHook' is defined in cfg
    #rfnext_init_model(model, cfg=cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)

    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)

        # In multi_gpu_test, if tmpdir is None, some tesnors
        # will init on cuda by default, and no device choice supported.
        # Init a tmpdir to avoid error on npu here.
        if cfg.device == 'npu' and args.tmpdir is None:
            args.tmpdir = './npu_tmpdir'

        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)

            coco_gt = dataset.coco  

            result_files = dataset.format_results(outputs)[0]
            coco_dt = coco_gt.loadRes(result_files['segm'])
            pr_results = compute_pr_metrics(coco_gt, coco_dt, iou_thrs=[0.5, 0.7, 0.9], iouType='segm')

            print(pr_results)

            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()
