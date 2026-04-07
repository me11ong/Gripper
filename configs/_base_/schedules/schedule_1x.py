#optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)
optimizer = dict(type='AdamW', lr=2.5e-4, weight_decay=0.05)

lr_config = dict(
    policy='CosineAnnealing',
    #min_lr=1e-6,
    min_lr=2.5e-6,
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1e-3,
    by_epoch=False
)

optimizer_config = dict(
    #grad_clip=dict(max_norm=1.0)
    grad_clip=None
)

