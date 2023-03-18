_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/utdac_detection.py',
    '../_base_/schedules/schedule_20e.py',
    '../_base_/default_runtime.py'
]

cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b0',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='BN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        # init_cfg=dict(
        #     type='Pretrained', prefix='backbone', checkpoint=checkpoint)
    ),
    neck=dict(
        in_channels=[40, 112, 320],
        start_level=0,
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg),
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [dict(type='RandomRotate90', always_apply=False, p=0.5)]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MixUp', p=0.5, lambd=0.5),
    dict(type='Resize',
         img_scale=[(1333, 640), (1333, 800)],
         multiscale_mode='range',
         keep_ratio=True),  # 采用多尺度训练
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='Albu',
         transforms=albu_train_transforms,
         bbox_params=dict(type='BboxParams',
                          format='pascal_voc',
                          label_fields=['gt_labels'],
                          min_visibility=0.0,
                          filter_lost_elements=True),
         keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
         update_pad_shape=False,
         skip_img_without_anno=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(4096, 600), (4096, 800), (4096, 1000)],  # 采用多尺度预测
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

evaluation = dict(interval=1, metric='bbox', classwise=True)
optimizer_config = dict(grad_clip=None)
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=5e-05,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])

auto_resume = False
gpu_ids = [0]
