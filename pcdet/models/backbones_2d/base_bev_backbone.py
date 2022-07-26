import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        # 读取下采样层参数
        # cfg.MODEL.BACKBONE_2D.LAYER_NUMS: [3, 5, 5]
        # cfg.MODEL.BACKBONE_2D.LAYER_STRIDES: [2, 2, 2]
        # cfg.MODEL.BACKBONE_2D.NUM_FILTERS: [64, 128, 256]
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []
        # 读取上采样层参数
        # cfg.MODEL.BACKBONE_2D.UPSAMPLE_STRIDES: [1, 2, 4]
        # cfg.MODEL.BACKBONE_2D.NUM_UPSAMPLE_FILTERS: [128, 128, 128]
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        # c_in_list:(64, 64, 128) 
        # input_channels:64, num_filters[:-1]：64,128
        c_in_list = [input_channels, *num_filters[:-1]] 
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            # (64,64)-->(64,128)-->(128,256)
            # 这里为cur_layers的第一层且stride=2
            cur_layers = [
                # 零填充函数是nn.ZeroPad2d，也就是对Tensor使用0进行边界填充
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            # layer_nums [3, 5, 5]
            for k in range(layer_nums[idx]):# 根据layer_nums堆叠卷积层
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            # 在block中添加该层
            # *作用是：将列表解开成几个独立的参数，传入函数 # 类似的运算符还有两个星号(**)，是将字典解开成独立的元素作为形参
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    # ConvTranspose2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    # BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                    # ReLU()
                    # ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2), bias=False)
                    # BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                    # ReLU()
                    # ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(4, 4), bias=False)
                    # BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                    # ReLU()
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) # 384
        if len(upsample_strides) > num_levels: # 3 = 3 
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features:(4,64,496,432)
        Returns:
        """
        # 不断缩小特征图的分辨率，同时提升特征图的维度,获得了三个不同分辨率的特征图
        # 另一个网络对三个特征图进行上采样至相同大小，然后进行concatenation
        spatial_features = data_dict['spatial_features'] # (4, 64, 496, 432)
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            # 下采样后：(4,64,248,216)-->(4,128,124,108)-->(4,256,62,54)
            x = self.blocks[i](x)
            # stride: (4, 64, 496, 432)/(4,64,248,216)  (4, 64, 496, 432)/(4,128,124,108)    (4, 64, 496, 432)/(4,256,62,54)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x)) # （4，128，248，216） （4，128，248，216） （4，128，248，216）
            else:
                ups.append(x)
        # 如果存在上采样层，将上采样结果连接
        if len(ups) > 1:
            x = torch.cat(ups, dim=1) # 拼接得到维度 （4，384，248，216）
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        # 将结果存储在spatial_features_2d中并返回
        data_dict['spatial_features_2d'] = x

        return data_dict
