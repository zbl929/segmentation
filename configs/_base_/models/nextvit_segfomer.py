# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='nextvit_biformer',
                  frozen_stages=-1,
                  norm_eval=False,
                  with_extra_norm=True,
                  norm_cfg=norm_cfg,
                  resume=None,),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[96, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=1.0,
                # class_weight=[0.8373, 1.555]
            ),
            dict(
                type='FocalLoss',
                loss_name='loss_focal',
                loss_weight=1.0,
                # class_weight=[0.8373, 1.555]
            ),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=1.0,
                # class_weight=[0.8373, 1.555]
            )]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))