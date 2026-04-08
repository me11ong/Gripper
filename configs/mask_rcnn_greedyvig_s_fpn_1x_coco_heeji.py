_base_ = [
    '_base_/models/mask_rcnn_r50_fpn_heeji.py',
    '_base_/datasets/coco_instance_heeji.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=1, 
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=999999,
        save_last=True,
        save_best='coco/segm_mAP',
        rule='greater',
    ),
    logger=dict(
        type='LoggerHook',
        interval=50, 
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)