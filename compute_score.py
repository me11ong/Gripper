import argparse
import numpy as np
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from pycocotools import mask as maskUtils
from tqdm import tqdm

import greedyvig_backbone_260217
import CustomMaskRCNN
import 참고_코드.GreedyViG.detection.CustomCOCODataset as CustomCOCODataset
import CustomTransform

# -----------------------------
# Argument
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='config file path')
parser.add_argument('--pkl', required=True, help='test result pkl file')
parser.add_argument('--score_thr', type=float, default=0.5)
args = parser.parse_args()


# -----------------------------
# Load dataset & results
# -----------------------------
cfg = Config.fromfile(args.config)
dataset = build_dataset(cfg.data.test)

results = mmcv.load(args.pkl)

num_classes = len(dataset.CLASSES)


# -----------------------------
# Metric functions
# -----------------------------
def compute_iou(pred, gt, num_classes):
    ious = []
    for cls in range(1, num_classes + 1):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            continue

        ious.append(intersection / union)

    return np.mean(ious) if len(ious) > 0 else 0


def compute_dice(pred, gt, num_classes):
    dices = []
    for cls in range(1, num_classes + 1):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        denom = pred_cls.sum() + gt_cls.sum()

        if denom == 0:
            continue

        dices.append(2 * intersection / denom)

    return np.mean(dices) if len(dices) > 0 else 0


# -----------------------------
# Convert prediction to semantic map
# (score 정렬 + threshold 적용)
# -----------------------------
def pred_to_semantic(bbox_results, segm_results, img_shape, score_thr):
    H, W = img_shape[:2]
    semantic_map = np.zeros((H, W), dtype=np.int32)

    all_instances = []

    # 모든 클래스에 대해 instance 수집
    for cls_id in range(len(segm_results)):
        bboxes = bbox_results[cls_id]
        masks = segm_results[cls_id]

        for i in range(len(bboxes)):
            score = bboxes[i][-1]

            if score < score_thr:
                continue

            all_instances.append({
                'cls_id': cls_id,
                'score': score,
                'mask': masks[i]
            })

    # score 기준 내림차순 정렬
    all_instances = sorted(all_instances, key=lambda x: x['score'], reverse=True)

    # 높은 score부터 덮어쓰기
    for inst in all_instances:
        decoded = maskUtils.decode(inst['mask'])
        semantic_map[decoded.astype(bool)] = inst['cls_id'] + 1

    return semantic_map


# -----------------------------
# GT to semantic map
# -----------------------------
# def gt_to_semantic(gt_masks, gt_labels, img_shape):
#     H, W = img_shape[:2]
#     semantic_map = np.zeros((H, W), dtype=np.int32)

#     for mask, label in zip(gt_masks.masks, gt_labels):
#         semantic_map[mask.astype(bool)] = label + 1

#     return semantic_map

def gt_to_semantic(gt_masks, gt_labels, img_shape, num_classes):
    """
    gt_masks : list (polygon or RLE)
    gt_labels: ndarray
    img_shape: (H, W)
    """
    H, W = img_shape
    semantic_map = np.zeros((H, W), dtype=np.int32)

    for mask_ann, label in zip(gt_masks, gt_labels):

        # polygon -> RLE 변환
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, H, W)
            rle = maskUtils.merge(rles)

        # 이미 RLE인 경우
        elif isinstance(mask_ann, dict):
            rle = mask_ann

        else:
            raise TypeError(f"Unknown mask type: {type(mask_ann)}")

        mask = maskUtils.decode(rle)

        semantic_map[mask == 1] = label + 1  # background=0 유지

    return semantic_map

# -----------------------------
# Main loop
# -----------------------------
mIoUs = []
mDices = []

for idx in tqdm(range(len(dataset))):

    bbox_results, segm_results = results[idx]

    ann = dataset.get_ann_info(idx)
    gt_masks = ann['masks']
    gt_labels = ann['labels']

    img_info = dataset.data_infos[idx]
    img_shape = (img_info.get('height'), img_info.get('width'))

    pred_map = pred_to_semantic(
        bbox_results,
        segm_results,
        img_shape,
        args.score_thr
        
    )

    gt_map = gt_to_semantic(
        gt_masks,
        gt_labels,
        img_shape,
        num_classes = len(dataset.CLASSES)
    )

    mIoUs.append(compute_iou(pred_map, gt_map, num_classes))
    mDices.append(compute_dice(pred_map, gt_map, num_classes))


print("\n==============================")
print("Final mIoU  :", np.mean(mIoUs))
print("Final mDice :", np.mean(mDices))
print("==============================")
