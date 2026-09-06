# harl/models/base/mlp.py

import torch.nn as nn
from harl.utils.models_tools import init, get_active_func, get_init_method

"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func):
        """Initialize the MLP layer.
        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list) list of hidden layer sizes.
            initialization_method: (str) initialization method.
            activation_func: (str) activation function.
        """
        super(MLPLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [
            init_(nn.Linear(input_dim, hidden_sizes[0])),
            active_func,
            nn.LayerNorm(hidden_sizes[0]),
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])),
                active_func,
                nn.LayerNorm(hidden_sizes[i]),
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLPBase(nn.Module):
    """A MLP base module."""

    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()

        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]

        obs_dim = obs_shape[0]

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func
        )

    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x

# ================= 【新增代码：结合 2025 论文的深度残差架构】 =================

class RLResBlock(nn.Module):
    """
    根据论文设计的强化学习残差块：
    包含 4 个重复单元：Linear -> LayerNorm -> Swish(SiLU)
    """
    def __init__(self, hidden_size, initialization_method):
        super(RLResBlock, self).__init__()
        
        init_method = get_init_method(initialization_method)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        layers = []
        for _ in range(4): # 论文指定每个 block 有 4 层
            layers.append(init_(nn.Linear(hidden_size, hidden_size)))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.SiLU()) # SiLU 就是 Swish 激活函数
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # 残差连接：输入 + 经过 4 层变换后的输出
        return x + self.block(x)


class DeepResNetBase(nn.Module):
    """
    支持极深网络 (Scaling Depth) 的基础特征提取器
    """
    def __init__(self, args, obs_shape):
        super(DeepResNetBase, self).__init__()
        
        self.use_feature_normalization = args.get("use_feature_normalization", True)
        self.initialization_method = args.get("initialization_method", "orthogonal_")
        
        # 为了兼容，如果传入的是 hidden_sizes 列表，我们取第一个元素作为统一隐藏层维度
        self.hidden_size = args["hidden_sizes"][0] 
        
        # 通过配置文件传入残差块的数量。例如 8 个 block = 32 层 dense
        self.num_res_blocks = args.get("num_res_blocks", 1)

        init_method = get_init_method(self.initialization_method)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # 1. 初始特征归一化
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_shape[0])
        else:
            self.feature_norm = nn.Identity()

        # 2. 将观测维度投影到 hidden_size
        self.input_proj = nn.Sequential(
            init_(nn.Linear(obs_shape[0], self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU()
        )

        # 3. 堆叠深层残差块
        blocks = []
        for _ in range(self.num_res_blocks):
            blocks.append(RLResBlock(self.hidden_size, self.initialization_method))
            
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.feature_norm(x)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return x