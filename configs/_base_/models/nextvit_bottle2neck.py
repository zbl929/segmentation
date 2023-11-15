# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='nextvit_bottle2neck',
                  frozen_stages=-1,
                  norm_eval=False,
                  with_extra_norm=True,
                  norm_cfg=norm_cfg,
                  resume=None,),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[96, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=512,
        ham_channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True),
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=3.0,
                class_weight=[0.8373, 1.555]
            ),
            dict(
                type='FocalLoss',
                loss_name='loss_focal',
                loss_weight=3.0,
                class_weight=[0.8373, 1.555]
            ),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=1.0,
                class_weight=[0.8373, 1.555]
            )]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))