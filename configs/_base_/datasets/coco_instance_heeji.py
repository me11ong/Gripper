# dataset settings
dataset_type = 'CustomCocoDataset'
data_root = '/mnt/E/Gripper/참고_코드/GreedyViG/detection/data/COCO/'

classes = ('person','bicycle','car','motorcycle','airplane',
            'bus','train','truck','boat','traffic light',
            'fire hydrant','stop sign','parking meter',
            'bench','bird','cat','dog','horse','sheep',
            'cow','elephant','bear','zebra','giraffe',
            'backpack','umbrella','handbag','tie','suitcase',
            'frisbee','skis','snowboard','sports ball',
            'kite','baseball bat','baseball glove','skateboard'
            ,'surfboard','tennis racket','bottle','wine glass',
            'cup','fork','knife','spoon','bowl','banana','apple',
            'sandwich','orange','broccoli','carrot','hot dog',
            'pizza','donut','cake','chair','couch','potted plant',
            'bed','dining table','toilet','tv','laptop','mouse',
            'remote','keyboard','cell phone','microwave','oven',
            'toaster','sink','refrigerator','book','clock','vase',
            'scissors','teddy bear','hair drier','toothbrush')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='LoadTextAnnotations'),

    dict(type='Resize', img_scale=(384, 384), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomCrop', crop_size=(192, 192)),
    dict(type='Pad', size=(384, 384)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks',],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'scale_factor','pad_shape',
        'img_norm_cfg','ori_filename', 'texts' )),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTextAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 384),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(384, 384)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'scale_factor','pad_shape',
        'img_norm_cfg','ori_filename', 'texts' )),
            
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        filter_empty_gt=True,
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/refcoco(unc)_train.json',
        img_prefix=data_root + 'train2014/',
        pipeline=train_pipeline,),

    val=dict(
        filter_empty_gt=True,
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/refcoco(unc)_valid.json',
        img_prefix=data_root + 'train2014/',
        pipeline=test_pipeline,),

    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/refcoco(unc)_testA.json',
        img_prefix=data_root + 'train2014/',
        pipeline=test_pipeline,))


