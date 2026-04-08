
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2.5e-4,
        weight_decay=0.05,
    ),
    clip_grad=None,             
)


param_scheduler = [
    
    dict(
        type='LinearLR',
        start_factor=1e-3,          
        by_epoch=False,
        begin=0,
        end=5000,                   
    ),

    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        begin=5000,
        end=None,                   
        eta_min=2.5e-6,             
    ),
]