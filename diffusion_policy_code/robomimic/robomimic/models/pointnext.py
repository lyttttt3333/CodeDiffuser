import torch.nn as nn
import torch


class SetAbstraction(nn.Module):
    """
    点云特征提取
    包含一个单尺度S-G-P过程
    """

    def __init__(self,
                 npoint: int,
                 radius: int,
                 nsample: int,
                 in_channel: int,
                 coor_dim: int = 3):
        """
        :param npoint: 采样点数量
        :param radius: 采样半径
        :param nsample: 采样点数量
        :param in_channel: 特征维度的输入值
        :param coor_dim: 点的坐标维度，默认为3
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.coor_dim = coor_dim
        self.mlp = build_mlp(in_channel=in_channel + coor_dim, channel_list=[in_channel * 2], dim=2)

    def forward(self, points_coor, points_fea):
        """
        :param points_coor: <torch.Tensor> (B, 3, N) 点云原始坐标
        :param points_fea: <torch.Tensor> (B, C, N) 点云特征
        :return:
            new_xyz: <torch.Tensor> (B, 3, S) 下采样后的点云坐标
            new_fea: <torch.Tensor> (B, D, S) 采样点特征
        """
        points_coor, points_fea = points_coor.permute(0, 2, 1), points_fea.permute(0, 2, 1)  # (B, C, N) -> (B, N, C)
        bs, nbr_point_in, _ = points_coor.shape
        num_point_out = self.npoint

        '''S 采样'''
        fps_idx = farthest_point_sample(points_coor, num_point_out)
        new_coor = index_points(points_coor, fps_idx)  # 获取新采样点 (B, S, coor)
        # new_attention = index_points(points_attention, fps_idx)

        '''G 分组'''
        # 每个group的点云索引 (B, S, K)
        group_idx = query_hybrid(self.radius, self.nsample, points_coor[..., :3], new_coor[..., :3])

        # 基于分组获取各组内点云坐标和特征，并进行拼接
        grouped_points_coor = index_points(points_coor[..., :3], group_idx)  # 每个group内所有点云的坐标 (B, S, K, 3)
        grouped_points_coor -= new_coor[..., :3].view(bs, num_point_out, 1, 3)  # 坐标转化为与采样点的偏移量 (B, S, K, 3)
        grouped_points_coor = grouped_points_coor / self.radius  # 相对坐标归一化
        grouped_points_fea = index_points(points_fea, group_idx)  # 每个group内所有点云的特征 (B, S, K, C)
        grouped_points_fea = torch.cat([grouped_points_fea, grouped_points_coor], dim=-1)  # 拼接坐标偏移量 (B, S, K, C+3)

        '''P 特征提取'''
        # (B, S, K, C+3) -> (B, C+3, K, S) -mlp-> (B, D, K, S) -pooling-> (B, D, S)
        grouped_points_fea = grouped_points_fea.permute(0, 3, 2, 1)  # 2d卷积作用于维度1
        grouped_points_fea = self.mlp(grouped_points_fea)
        new_points_fea = torch.max(grouped_points_fea, dim=2)[0]

        new_coor = new_coor.permute(0, 2, 1)  # (B, 3, S)
        return new_coor, new_points_fea


class LocalAggregation(nn.Module):
    """
    局部特征提取
    包含一个单尺度G-P过程，每一个点都作为采样点进行group以聚合局部特征，无下采样过程
    """

    def __init__(self,
                 radius: int,
                 nsample: int,
                 in_channel: int,
                 coor_dim: int = 3):
        """
        :param radius: 采样半径
        :param nsample: 采样点数量
        :param in_channel: 特征维度的输入值
        :param coor_dim: 点的坐标维度，默认为3
        """
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.mlp = build_mlp(in_channel=in_channel + coor_dim, channel_list=[in_channel], dim=2)

    def forward(self, points_coor, points_fea):
        """
        :param points_coor: <torch.Tensor> (B, 3, N) 点云原始坐标
        :param points_fea: <torch.Tensor> (B, C, N) 点云特征
        :return:
            new_fea: <torch.Tensor> (B, D, N) 局部特征聚合后的特征
        """
        # (B, C, N) -> (B, N, C)
        points_coor, points_fea = points_coor.permute(0, 2, 1), points_fea.permute(0, 2, 1)
        bs, npoint, _ = points_coor.shape

        '''G 分组'''
        # 每个group的点云索引 (B, N, K)
        group_idx = query_hybrid(self.radius, self.nsample, points_coor[..., :3], points_coor[..., :3])

        # 基于分组获取各组内点云坐标和特征，并进行拼接
        grouped_points_coor = index_points(points_coor[..., :3], group_idx)  # 每个group内所有点云的坐标 (B, N, K, 3)
        grouped_points_coor = grouped_points_coor - points_coor[..., :3].view(bs, npoint, 1, 3)  # 坐标转化为与采样点的偏移量
        grouped_points_coor = grouped_points_coor / self.radius  # 相对坐标归一化
        grouped_points_fea = index_points(points_fea, group_idx)  # 每个group内所有点云的特征 (B, N, K, C)
        grouped_points_fea = torch.cat([grouped_points_fea, grouped_points_coor], dim=-1)  # 拼接坐标偏移量 (B, N, K, C+3)

        '''P 特征提取'''
        # (B, N, K, C+3) -> (B, C+3, K, N) -mlp-> (B, D, K, N) -pooling-> (B, D, N)
        grouped_points_fea = grouped_points_fea.permute(0, 3, 2, 1)  # 2d卷积作用于维度1
        grouped_points_fea = self.mlp(grouped_points_fea)
        new_fea = torch.max(grouped_points_fea, dim=2)[0]

        return new_fea


class InvResMLP(nn.Module):
    """
    逆瓶颈残差块
    """

    def __init__(self,
                 radius: int,
                 nsample: int,
                 in_channel: int,
                 coor_dim: int = 3,
                 expansion: int = 4):
        """
        :param radius: 采样半径
        :param nsample: 采样点数量
        :param in_channel: 特征维度的输入值
        :param coor_dim: 点的坐标维度，默认为3
        :param expansion: 中间层通道数扩张倍数
        """
        super().__init__()
        self.la = LocalAggregation(radius=radius, nsample=nsample, in_channel=in_channel, coor_dim=coor_dim)
        channel_list = [in_channel * expansion, in_channel]
        self.pw_conv = build_mlp(in_channel=in_channel, channel_list=channel_list, dim=1, drop_last_act=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points):
        """
        :param points:
            <torch.Tensor> (B, 3, N) 点云原始坐标
            <torch.Tensor> (B, C, N) 点云特征
        :return:
            new_fea: <torch.Tensor> (B, D, N)
        """
        points_coor, points_fea = points
        identity = points_fea
        points_fea = self.la(points_coor, points_fea)
        points_fea = self.pw_conv(points_fea)
        points_fea = points_fea + identity
        points_fea = self.act(points_fea)
        return [points_coor, points_fea]


class Stage(nn.Module):
    """
    PointNeXt一个下采样阶段
    """

    def __init__(self,
                 npoint: int,
                 radius_list: list,
                 nsample_list: list,
                 in_channel: int,
                 coor_dim: int = 3,
                 expansion: int = 4):
        """
        :param npoint: 采样点数量
        :param radius_list: <list[float]> 采样半径
        :param nsample_list: <list[int]> 采样邻点数量
        :param in_channel: 特征维度的输入值
        :param coor_dim: 点的坐标维度，默认为3
        :param expansion: 中间层通道数扩张倍数
        """
        super().__init__()
        self.sa = SetAbstraction(npoint=npoint, radius=radius_list[0], nsample=nsample_list[0],
                                 in_channel=in_channel, coor_dim=coor_dim)

        irm = []
        for i in range(1, len(radius_list)):
            irm.append(
                InvResMLP(radius=radius_list[i], nsample=nsample_list[i], in_channel=in_channel * 2,
                          coor_dim=coor_dim, expansion=expansion)
            )
        self.irm = nn.Sequential(*irm)

    def forward(self, points_coor, points_fea):
        """
        :param points_coor: <torch.Tensor> (B, 3, N) 点云原始坐标
        :param points_fea: <torch.Tensor> (B, D, N) 点云特征
        :return:
            new_xyz: <torch.Tensor> (B, 3, S) 下采样后的点云坐标
            new_points_concat: <torch.Tensor> (B, D', S) 下采样后的点云特征
        """
        new_coor, new_points_fea = self.sa(points_coor, points_fea)
        new_coor, new_points_fea = self.irm([new_coor, new_points_fea])
        return new_coor, new_points_fea

def build_mlp(in_channel, channel_list, dim=2, bias=False, drop_last_act=False,
              drop_last_norm_act=False, dropout=False):
    """
    构造基于n dim 1x1卷积的mlp
    :param in_channel: <int> 特征维度的输入值
    :param channel_list: <list[int]> mlp各层的输出通道维度数
    :param dim: <int> 维度，1或2
    :param bias: <bool> 卷积层是否添加bias，一般BN前的卷积层不使用bias
    :param drop_last_act: <bool> 是否去除最后一层激活函数
    :param drop_last_norm_act: <bool> 是否去除最后一层标准化层和激活函数
    :param dropout: <bool> 是否添加dropout层
    :return: <torch.nn.ModuleList[torch.nn.Sequential]>
    """
    # 解析参数获取相应卷积层、归一化层、激活函数
    if dim == 1:
        Conv = nn.Conv1d
        NORM = nn.BatchNorm1d
    else:
        Conv = nn.Conv2d
        NORM = nn.BatchNorm2d
    ACT = nn.ReLU

    # 根据通道数构建mlp
    mlp = []
    for i, channel in enumerate(channel_list):
        if dropout and i > 0:
            mlp.append(nn.Dropout(0.5, inplace=False))
        # 每层为conv-bn-relu
        mlp.append(Conv(in_channels=in_channel, out_channels=channel, kernel_size=1, bias=bias))
        # mlp.append(NORM(channel))
        mlp.append(ACT(inplace=True))
        if i < len(channel_list) - 1:
            in_channel = channel

    if drop_last_act:
        mlp = mlp[:-1]
    elif drop_last_norm_act:
        mlp = mlp[:-2]
        mlp[-1] = Conv(in_channels=in_channel, out_channels=channel, kernel_size=1, bias=True)

    return nn.Sequential(*mlp)


def coordinate_distance(src, dst):
    """
    计算两个点集的各点间距
    !!!使用半精度运算或自动混合精度时[不要]使用化简的方法，否则会出现严重的浮点误差
    :param src: <torch.Tensor> (B, M, C) C为坐标
    :param dst: <torch.Tensor> (B, N, C) C为坐标
    :return: <torch.Tensor> (B, M, N)
    """
    B, M, _ = src.shape
    _, N, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).view(B, M, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, N)

    # dist = torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)).pow(2), dim=-1)
    return dist


def index_points(points, idx):
    """
    跟据采样点索引获取其原始点云xyz坐标等信息
    :param points: <torch.Tensor> (B, N, 3+) 原始点云
    :param idx: <torch.Tensor> (B, S)/(B, S, G) 采样点索引，S为采样点数量，G为每个采样点grouping的点数
    :return: <torch.Tensor> (B, S, 3+)/(B, S, G, 3+) 获取了原始点云信息的采样点
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    最远点采样
    随机选择一个初始点作为采样点，循环的将与当前采样点距离最远的点当作下一个采样点，直至满足采样点的数量需求
    :param xyz: <torch.Tensor> (B, N, 3+) 原始点云
    :param npoint: <int> 采样点数量
    :return: <torch.Tensor> (B, npoint) 采样点索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10  # 每个点与最近采样点的最小距离
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 随机选取初始点

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, -1)  # [bs, 1, coor_dim]
        dist = torch.nn.functional.pairwise_distance(xyz, centroid)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_hybrid(radius, nsample, xyz, new_xyz):
    """
    基于采样点进行KNN与ball query混合的grouping
    :param radius: <float> grouping半径
    :param nsample: <int> group内点云数量
    :param xyz: <torch.Tensor> (B, N, 3) 原始点云
    :param new_xyz: <torch.Tensor> (B, S, 3) 采样点
    :return: <torch.Tensor> (B, S, nsample) 每个采样点grouping的点云索引
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    dist = coordinate_distance(new_xyz, xyz)  # 每个采样点与其他点的距离的平方
    dist, group_idx = torch.topk(dist, k=nsample, dim=-1, largest=False)  # 基于距离选择最近的作为采样点
    radius = radius ** 2
    mask = dist > radius  # 距离较远的点替换为距离最近的点
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    group_idx[mask] = group_first[mask]

    return group_idx



class Head(nn.Module):
    """分类头 & 分割头"""
    def __init__(self, in_channel, mlp, num_class, task_type):
        """
        :param in_channel: <int> 特征维度的输入值
        :param mlp: <list[int]> mlp的通道维度数
        :param num_class: <int> 输出类别的数量
        """
        super(Head, self).__init__()
        mlp.append(num_class)
        self.mlp_modules = build_mlp(in_channel=in_channel, channel_list=mlp, dim=1,
                                     drop_last_norm_act=True, dropout=True)
        self.task_type = task_type

    def forward(self, points_fea):
        """
        :param points_fea: <torch.Tensor> (B, C, N) 点云特征
        :return: <torch.Tensor> (B, num_class, N) 点云特征
        """
        if self.task_type == 'classification':
            points_fea = torch.max(points_fea, dim=-1, keepdim=True)[0]  # (B, C, N) -> (B, C, 1)
        points_cls = self.mlp_modules(points_fea)
        return points_cls

class PointNeXt(nn.Module):

    def __init__(self, in_channels = 4, use_bn =False):
        super().__init__()
        self.type = 'classification'
        self.num_class = 256
        self.coor_dim = 3
        self.normal = 1
        width = 32
        self.npoint = [512, 128, 32, 8]
        self.radius_list=[[0.1, 0.2], [0.2, 0.4, 0.4], [0.4, 0.8], [0.8, 1.6]]
        self.nsample_list = [[16, 16], [16, 16, 16], [16, 16], [8, 8]]
        self.expansion=4
        self.head=[512, 256]

        self.mlp = nn.Conv1d(in_channels=self.coor_dim + self.normal,
                             out_channels=width, kernel_size=1)
        self.stage = nn.ModuleList()

        for i in range(len(self.npoint)):
            self.stage.append(
                Stage(
                    npoint=self.npoint[i], radius_list=self.radius_list[i], nsample_list=self.nsample_list[i],
                    in_channel=width, expansion=self.expansion, coor_dim=self.coor_dim
                )
            )
            width *= 2

        self.head = Head(in_channel=width, mlp=self.head, num_class=self.num_class, task_type=self.type)

    def forward(self, x):
        l0_xyz, l0_points = x[:, :self.coor_dim, :], x[:, :self.coor_dim + self.normal, :]
        l0_points = self.mlp(l0_points)

        attn = l0_points

        record = [[l0_xyz, l0_points]]
        for stage in self.stage:
            record.append(list(stage(*record[-1])))

        points_cls = self.head(record[-1][1])

        return points_cls.squeeze(-1)
    

if __name__ == '__main__':
    data = torch.rand(64, 4, 4000).to("cuda:0")
    model = PointNeXt().to("cuda:0")
    out = model(data)
    print(out.shape)



