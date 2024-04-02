_base_ = [
    '../_base_/datasets/my_dataset.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        # _delete_=True, # 未继承不用delete
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 2, 1, 2, 1),
        #dilations=(1, 1, 1, 2, 2, 4, 4),
        dilations=(1, 1, 1, 1, 1, 1, 1),
        out_indices=(1, 2, 4, 6),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    decode_head=dict(
        #type='DepthwiseSeparableASPPHead',
        type='DenseASPPHead',
        num_classes=6,
        in_channels=320,
        in_index=3,
        channels=512,
        # dilations=(1, 12, 24, 36),
        c1_in_channels=24,
        c1_channels=48,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        #loss_decode=[dict(type='Loss', loss_name='loss_ce', loss_weight=0.4),
        #            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)]),
        loss_decode=dict(type='CrossEntropyLoss', 
                        class_weight=[0.8553, 1.022, 1.018 ,1.028 ,1.096, 0.979]
                        # class_weight=[0.1, 0.2, 0.1 ,0.2 ,0.3, 0.1])
                        )),    
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# 设置工作路径，防止checkpoint被覆盖
# load_from = None 加载预训练模型，默认不加载

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005) # 可能初始学习率太小，还没有预训练权重
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        #power=0.9,
        begin=0,
        end=60000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=1000)# 50个iter就val能训练出啥
work_dir = './work_dirs/mobilenetv2dense'
