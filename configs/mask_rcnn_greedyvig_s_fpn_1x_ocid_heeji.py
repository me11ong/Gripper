_base_ = [
    '_base_/models/mask_rcnn_r50_fpn_heeji.py',
    '_base_/datasets/ocid_instance_heeji.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]

total_epochs = 150
checkpoint_config = dict(
    interval=999999,
    save_last=True, 
)

evaluation = dict(
    interval=1,
    metric=['bbox', 'segm'],
    save_best='segm_mAP'
)

log_config = dict(
    interval=50,   # 몇 iteration마다 로그 찍을지
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

