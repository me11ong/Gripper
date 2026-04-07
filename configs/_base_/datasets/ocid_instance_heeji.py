# dataset settings
dataset_type = 'CustomCocoDataset'
data_root = '/home/user/Gripper/data/OCID-VLG/'

classes = ('apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl',
            'cereal_box', 'coffee_mug', 'flashlight', 'food_bag', 'food_box',
            'food_can', 'glue_stick', 'hand_towel', 'instant_noodles',
            'keyboard', 'kleenex', 'lemon', 'lime', 'marker', 'orange',
            'peach', 'pear', 'potato', 'shampoo', 'soda_can', 'sponge',
            'stapler', 'tomato', 'toothpaste')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
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
    samples_per_gpu=16,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'ref_train.json',
        img_prefix=data_root + 'imgs_all',
        pipeline=train_pipeline,),

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'ref_valid.json',
        img_prefix=data_root + 'imgs_all',
        pipeline=test_pipeline,),

    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'ref_test.json',
        img_prefix=data_root + 'imgs_all',
        pipeline=test_pipeline,))


