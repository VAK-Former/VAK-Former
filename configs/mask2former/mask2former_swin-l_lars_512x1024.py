_base_ = [
    '../_base_/default_runtime.py',
]
custom_imports = dict(
    imports=[
        'mmseg.datasets.lars_dataset',
        'mmseg.evaluation.metrics.f1_metric'
    ],
    allow_failed_imports=False
)
load_from = r'E:\USV\mmsegmentation\work_dirs\mask2former_swin-l_lars_512x1024\best_mIoU_iter_4554.pth'

crop_size = (512, 1024)
ignore_idx = 255

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=ignore_idx,
    size=crop_size
)

num_classes = 3

depths = [2, 2, 18, 2]                        #pchange

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer',             
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],                 #pchange
        num_heads=[6, 12, 24, 48],            #pchange
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.4,                   #changed
        patch_norm=True,
        #layer_scale_init_value=1e-5,          #added
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
    ),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[192, 384, 768, 1536],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=200,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)
                    )
                ),
                init_cfg=None
            ),
            positional_encoding=dict(
                num_feats=128, normalize=True
            ),
            init_cfg=None
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            num_feats=128, normalize=True
        ),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True
                ),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True
                )
            ),
            init_cfg=None
        ),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]
        ),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0
        ),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0
        ),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]
            ),
            sampler=dict(type='mmdet.MaskPseudoSampler')
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(
    mode='slide',
    crop_size=crop_size,         # (512, 1024)
    stride=(341, 341),
    multi_scale=[0.75, 1.0, 1.25],
    flip=True)
)

# --- Dataset Config ---

data_root = r'E:\LARS\Dataset'
dataset_type = 'LaRSDataset'

# Train pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),  # set True or False based on your needs
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(
    # type='Albu',
    # transforms=[
    #     dict(type='RandomBrightnessContrast', p=0.5),
    #     dict(type='GaussBlur', p=0.3),
    # ],
    # keymap=dict(img='image', gt_semantic_seg='mask'),
    # update_pad_shape=False,
    # skip_img_without_anno=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', mean=data_preprocessor['mean'], std=data_preprocessor['std'], to_rgb=True),
    dict(type='Pad', size=crop_size, pad_val=0),
    dict(type='PackSegInputs'),
]


# Test
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(
    type='Normalize',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
    ),
    dict(type='Pad', size_divisor=32),
    dict(type='PackSegInputs')
]

# validation pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='MultiScaleFlipAug',
        transforms=[
            # dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='RandomFlip', prob=0.0),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='PackSegInputs')
        ]
    )
    # dict(type='Normalize', mean=data_preprocessor['mean'], std=data_preprocessor['std'], to_rgb=True),
    # dict(type='PackSegInputs')
]

val_pipeline = test_pipeline

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='lars_v1.0.0_images/train/images',
            seg_map_path='lars_v1.0.0_annotations/train/semantic_masks',
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='lars_v1.0.0_images/val/images',
            seg_map_path='lars_v1.0.0_annotations/val/semantic_masks',
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        # pipeline=test_pipeline,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='lars_v1.0.0_images/test/images',
        ),
        img_suffix='.jpg',
        pipeline=test_pipeline,
    )
)

val_evaluator = dict(
    type='IoUF1Metric',
    iou_metrics=['mIoU']
)
test_evaluator = val_evaluator

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    dataset=data['train']
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LarsDataset',
        data_root=...,
        img_dir=...,
        ann_dir=...,
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LarsDataset',
        data_root=...,
        img_dir=...,
        ann_dir=...,
        pipeline=test_pipeline   # ✅ THIS IS THE KEY LINE
    )
)

# rest of config remains the same
# optimizer, lr scheduler, hooks, etc.

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})

optimizer = dict(
    type='AdamW', lr=0.00003, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.5, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0)
)


param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0, power=0.9, begin=500, end=15000, by_epoch=False)
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=15000, val_interval=99
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop', compute_fps=True)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=99, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=99,
        save_best='mIoU'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

custom_hooks = [
    dict(
        type='EMAHook',
        momentum=0.0001,          # EMA decay = 0.9999
        update_buffers=True
    )
]

auto_scale_lr = dict(enable=False, base_batch_size=2)
