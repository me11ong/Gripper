import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import numpy as np
import cv2
import json
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import ProgressBar

from mmdet.utils import setup_cache_size_limit_of_dynamo

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from sklearn.metrics import confusion_matrix

import greedyvig_backbone_260313
import CustomMaskRCNN
import CustomCOCODataset
import CustomTransform


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
    (111, 7, 86)
]


def compute_pr_metrics(coco_gt, coco_dt, iou_thrs=[0.5, 0.7, 0.9], iouType='bbox'):
    cocoEval = COCOeval(coco_gt, coco_dt, iouType)
    cocoEval.params.iouThrs = np.array(iou_thrs)
    cocoEval.evaluate()
    cocoEval.accumulate()

    precision = cocoEval.eval['precision']
    results = {}
    for i, thr in enumerate(iou_thrs):
        pr = precision[i, :, :, 0, -1]
        pr = pr[pr > -1]
        results[f'PR@{int(thr * 100)}'] = np.mean(pr)

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
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            text = f'{cls_id}:{cls_name}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def compute_metrics(all_predictions, all_ground_truths, num_classes):
    """
    confusion matrix 기반 mACC, mIoU, fwIoU, pACC, mDice 계산
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

    Dice = (2 * TP) / (2 * TP + FP + FN + 1e-10)
    mDice = np.nanmean(Dice)

    return {
        "mACC": mACC,
        "mIoU": mIoU,
        "fwIoU": fwIoU,
        "pACC": pACC,
        'mDice': mDice
    }


def polygons_to_mask(polygons, height, width):
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)
    return mask.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use (only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation.')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, e.g., "bbox", "segm"')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--show-score-thr', type=float, default=0.5)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    return args


def main():
    args = parse_args()

    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)


    if args.work_dir is not None:
        cfg.work_dir = args.work_dir


    cfg.load_from = args.checkpoint

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True


    runner = Runner.from_cfg(cfg)

    load_checkpoint(runner.model, args.checkpoint, map_location='cpu')

    runner.test()

    if args.eval and 'segm' in args.eval:
        print("\n[INFO] Computing additional PR metrics...")
        ann_file = cfg.test_dataloader.dataset.ann_file
        coco_gt = COCO(ann_file)

        result_file = osp.join(cfg.work_dir, 'results.segm.json')
        if osp.exists(result_file):
            coco_dt = coco_gt.loadRes(result_file)
            pr_results = compute_pr_metrics(coco_gt, coco_dt, iou_thrs=[0.5, 0.7, 0.9], iouType='segm')
            print(pr_results)


if __name__ == '__main__':
    main()