import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer # pointpillar为true
        # 根据论文中，这是是简化版pointnet网络层的初始化
        # 论文中使用的是 1x1 的卷积层完成这里的升维操作（理论上使用卷积的计算速度会更快）
        # 输入的通道数是刚刚经过数据增强过后的点云特征，每个点云有10个特征，
        # 输出的通道数是64
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            # x的维度由（M, 32, 10）升维成了（M, 32, 64）
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        # BatchNorm1d层:(M, 64, 32) --> (M, 32, 64)
        # （pillars,num_point,channel）->(pillars,channel,num_points)
        # 这里之所以变换维度，是因为BatchNorm1d在通道维度上进行,对于图像来说默认模式为[N,C,H*W],通道在第二个维度上
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        # 完成pointnet的最大池化操作，找出每个pillar中最能代表该pillar的点
        # x_max shape ：（M, 1, 64）
        x_max = torch.max(x, dim=1, keepdim=True)[0] # 这里的0是表示取数值，max的1表示索引
        # 返回经过简化版pointnet处理pillar的结果
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        """
        model_cfg:NAME: PillarVFE
                        WITH_DISTANCE: False
                        USE_ABSLOTE_XYZ: True
                        USE_NORM: True
                        NUM_FILTERS: [64]
        num_point_features:4
        voxel_size:[0.16 0.16 4]
        POINT_CLOUD_RANGE: [0, -39.68, -3, 69.12, 39.68, 1]
        """
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM # True
        self.with_distance = self.model_cfg.WITH_DISTANCE # False
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ # True
        # 如果use_absolute_xyz==True，则num_point_features=4+6，否则为3
        # x,y,z, x c , y c , z c , x p 和 y p（注在openpcdet的代码实现中是10维，多了一个zp，也就是该点在z轴上与该点所处pillar的z轴中心的偏移量）
        num_point_features += 6 if self.use_absolute_xyz else 3
        # 如果使用距离特征，即使用sqrt(x^2+y^2+z^2)，则使用特征加1
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
        # 加入线性层，将10维特征变为64维特征
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        # nn.ModuleList 搭建基础网络的时候，存储不同module即神经网络中的Linear、conv2d、Relu层等
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] # 0.16/2 + 0 = 0.08
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] # 0.16/2 + (-39.68) = -39.6
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] # 4/2 + (-3) = -1

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        计算padding的指示
        Args:
            actual_num:每个voxel实际点的数量（M，）
            max_num:voxel最大点的数量（32，）
        Returns:
            paddings_indicator:表明一个pillar中哪些是真实数据，哪些是填充的0数据
        """
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        """
        batch_dict:
            points:(97687,5)
            frame_id:(4,) --> (2238,2148,673,593)
            gt_boxes:(4,40,8)--> (x,y,z,dx,dy,dz,ry,class)
            use_lead_xyz:(4,) --> (1,1,1,1)
            voxels:(M,32,4) --> (x,y,z,intensity)
            voxel_coords:(M,4) --> (batch_index,z,y,x) 在dataset.collate_batch中增加了batch索引
            voxel_num_points:(M,)
            image_shape:(4,2) [[375 1242],[374 1238],[375 1242],[375 1242]]
            batch_size:4
        """
        # 假设一份点云数据为N*4,那么经过pillar生成后得到三份数据
        # voxel_features代表了每个生成的跑pillar数据，维度为[M，32,4]
        # coords代表了每个生成的pillar的zyx坐标，维度为[M,3]，其中z为0
        # voxel_num_points代表了每个生成的pillar中有多少个有效的点，维度是[m,]，因为不满32用0填充
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # 求每个voxle的平均值(M, 1, 3)--> (M, 1, 3) / (M, 1, 1)
        # 被求和的维度，在求和后会变为1，如果没有keepdim=True的设置，python会默认压缩该维度，比如变为[M, 3]
        # view扩充维度
        # 每个pillar点云的平均值
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1) # (M, 1, 3)
         # 每个点云减去该点对应pillar平均值得到差值xc,yc,zc
        f_cluster = voxel_features[:, :, :3] - points_mean # (M, 32, 3) 

        f_center = torch.zeros_like(voxel_features[:, :, :3]) # (M, 32, 3)
        #  (M, 32) - (M, 1)[(M,)-->(M, 1)] 
        #  coords是网格点坐标，不是实际坐标，乘以voxel大小再加上偏移量是恢复网格中心点实际坐标
        # 创建每个点到该pillar几何中心的偏移量xp,yp,zp
        # 此处多了z偏移，论文中没有

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        # 如果使用绝对坐标，直接组合
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        # 否则，取voxel_features的3维之后，在组合
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        # 如果使用距离信息
        if self.with_distance:
            # torch.norm的第一个2指的是求2范数，第二个2是在第三维度求范数
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True) # (M, 32, 1)
            features.append(points_dist)
        # 将特征在最后一维度拼接
        features = torch.cat(features, dim=-1) # （M，32，10）--> 10 = 4 +3 +3

        voxel_count = features.shape[1] # 32
        # 由于在生成每个pillar中，不满足最大32个点的pillar会由0填充，而刚才上面的计算出的xc,yc,zc和xp,yp,zp出现数值，需要将这些被填充的数据清0
        # get_paddings_indicator计算features中哪些是需要被保留真实数据和需要被置0的填充数据
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0) # (M, 32)
        # unsqueeze 增加维度 （M， 32）->(M, 32, 1)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features) # (M, 32, 1) 
        #(M，32,10）* (M, 32, 1)
        features *= mask #（M，32，10)--> 每个voxel未必填满，将被填充数据的所有特征置0
        for pfn in self.pfn_layers:
            # (M,32,10)-->(M,32,64)-->(M,1,64)
            features = pfn(features) # conv+BatchNorm+relu+maxpool
        # (M, 64), 每个pillar抽象出一个64维特征
        features = features.squeeze() # 缩减维度
        batch_dict['pillar_features'] = features
        return batch_dict
