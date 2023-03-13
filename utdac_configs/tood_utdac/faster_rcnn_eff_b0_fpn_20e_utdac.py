_base_ = [
    '../_base_/datasets/utdac_detection.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth'  # noqa

# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='EfficientNet',
        arch='b0',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='BN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    neck=dict(
        type='FPN',
        in_channels=[40, 112, 320],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

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
