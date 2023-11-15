# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='nextvit_dat',
                  frozen_stages=-1,
                  norm_eval=False,
                  with_extra_norm=True,
                  norm_cfg=norm_cfg,
                  resume=None,),
    decode_head=dict(
        type='LightHead_new',
        in_channels=[96, 256, 512, 1024],
        in_index=[0, 1, 2,3],
        channels=256,
        dropout_ratio=0.1,
        embed_dims=[256, 256, 256],
        num_classes=4,
        is_dw=True,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=1.0,
                class_weight=[0.8573, 1,1.155]
            ),
            dict(
                type='FocalLoss',
                loss_name='loss_focal',
                loss_weight=1.0,
                class_weight=[0.8573, 1,1.155]
            ),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=1.0,
                class_weight=[0.8573,1, 1.155]
             )
        ]),
    # auxiliary_head=dict(
    #     type='SegformerHead',
    #     in_channels=[96, 256],
    #     in_index=[0, 1],
    #     channels=128,
    #     dropout_ratio=0.1,
    #     num_classes=4,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=[
    #         dict(
    #             type='DiceLoss',
    #             loss_name='loss_dice',
    #             loss_weight=1.0,
    #             class_weight=[0.8573,1, 1.155, 1.155]
    #         )]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))