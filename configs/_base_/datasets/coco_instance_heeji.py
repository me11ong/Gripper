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
            'kite','baseball bat','baseball glove','skateboard',
            'surfboard','tennis racket','bottle','wine glass',
            'cup','fork','knife','spoon','bowl','banana','apple',
            'sandwich','orange','broccoli','carrot','hot dog',
            'pizza','donut','cake','chair','couch','potted plant',
            'bed','dining table','toilet','tv','laptop','mouse',
            'remote','keyboard','cell phone','microwave','oven',
            'toaster','sink','refrigerator','book','clock','vase',
            'scissors','teddy bear','hair drier','toothbrush')


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='LoadTextAnnotations'),                           
    dict(type='Resize', scale=(384, 384), keep_ratio=True),    
    dict(type='RandomFlip', prob=0.5),                        
    dict(type='RandomCrop', crop_size=(192, 192)),
    dict(type='Pad', size=(384, 384)),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='PackDetInputs',                                   
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'pad_shape', 'flip', 'flip_direction',
            'ori_filename', 'texts',                           
        ),
    ),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTextAnnotations'),                          
    dict(type='Resize', scale=(384, 384), keep_ratio=True),
    dict(type='Pad', size=(384, 384)),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'pad_shape',
            'ori_filename', 'texts',
        ),
    ),
]


train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True),
        ann_file=data_root + 'annotations/refcoco(unc)_train.json',
        data_prefix=dict(img=data_root + 'train2014/'),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True),
        ann_file=data_root + 'annotations/refcoco(unc)_valid.json',
        data_prefix=dict(img=data_root + 'train2014/'),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        ann_file=data_root + 'annotations/refcoco(unc)_testA.json',
        data_prefix=dict(img=data_root + 'train2014/'),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/refcoco(unc)_valid.json',
    metric=['bbox', 'segm'],
    format_only=False,
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/refcoco(unc)_testA.json',
    metric=['bbox', 'segm'],
    format_only=False,
)