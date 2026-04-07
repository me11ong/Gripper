_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        save_best="coco/bbox_mAP",
        rule="greater"
    )
)
model = dict(
    backbone=dict(
        type='greedyvig_s_feat',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/D/LAB/Gripper/참고_코드/GreedyViG/detection/Results/model/GreedyViG_S_Det.pth',
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        num_outs=5))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)

data = dict(
    samples_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='/mnt/D/LAB/Gripper/mmdetection/data/coco/annotations_trainval2017/annotations/instances_train2017.json',
        img_prefix='/mnt/D/LAB/Gripper/mmdetection/data/coco/train2017/',
    ),
    val=dict(
        type='CocoDataset',
        ann_file='/mnt/D/LAB/Gripper/mmdetection/data/coco/annotations_trainval2017/annotations/instances_val2017.json',
        img_prefix='/mnt/D/LAB/Gripper/mmdetection/data/coco/train2017/',
    )
)
