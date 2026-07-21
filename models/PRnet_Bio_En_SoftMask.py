# -*- coding: utf-8 -*-
# @Description: Soft Mask 版本的 PRnet_Bio（参照 expiMap, Lotfollahi et al., Nat Cell Biol 2023）
#               将 Hard Mask 改为 Soft Mask，通过 L1 正则化代替直接清零
# @Based on:    models/PRnet_Bio.py（原始 Hard Mask 版本）

from xmlrpc.client import Boolean
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F 
from torch.autograd import Variable
from models.PRnet import PRnet, PGM, PDecoder, PEncoder, PAdaptor # 导入原始组件

import anndata
from anndata import AnnData
from typing import Optional, Union

from scipy import sparse
import scanpy as sc

# =============================================================================
# PRnetBio（Soft Mask 版本）
# 继承自原始 PRnet，仅替换 PGM 中的 Decoder 为 Soft Mask 版本
# =============================================================================
class PRnetBio(PRnet):
    def __init__(self, adata, **kwargs):
        mask = kwargs.pop('mask', None)
        super().__init__(adata, **kwargs)
        # 动态替换 PGM
        self.PGM = PGMBio(self.x_dim_, self.c_dim, self.n_dim, self.hidden_layer_sizes_, self.z_dim_, self.adaptor_layer_sizes, self.comb_adapt_dim, self.dr_rate_, mask=mask)
        

class PGMBio(PGM):
    def __init__(self, x_dim, c_dim, n_dim, hidden_layer_sizes, z_dimension, adaptor_layer_sizes, comb_adapt_dim, dr_rate, mask=None):
        # 1. 运行原有的所有初始化 (建立 Encoder, Adaptor 等)
        super().__init__(x_dim, c_dim, n_dim, hidden_layer_sizes, z_dimension, adaptor_layer_sizes, comb_adapt_dim, dr_rate)
        
        # 2. 准备新解码器的层维度参数
        decoder_layer_sizes = hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        # 原本代码里最后会 append(x_dim*2)，PDecoderBio 内部逻辑会处理这个
        
        # 3. 唯独替换掉 decoder 这一项，传入你准备好的生物学 mask
        # 这里的第一个参数 z_dimension + c_dim + n_dim 确保了输入的隐变量维度正确
        self.decoder = PDecoderBio(
            z_dimension + c_dim + n_dim, 
            decoder_layer_sizes, 
            x_dim, 
            dr_rate, 
            mask=mask
        )


class PEncoder(nn.Module):
    """
    Constructs the  Perturb-encoder. This class implements the encoder part of PRnet. It will transform primary data in the `n_vars` dimension-space and chemical perturbation to `z_dimension` latent space.

    """

    def __init__(self, layer_sizes: list, z_dimension: int, dropout_rate: float):
        super().__init__() # to run nn.Module's init method

        # encoder architecture
        self.FC = None
        if len(layer_sizes) > 1:
            print("Encoder Architecture:")
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    print("\tInput Layer in, out:", in_size, out_size)
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
                else:
                    print("\tHidden Layer", i, "in/out:", in_size, out_size)
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                    self.FC.add_module("N{:d}".format(i), module=nn.BatchNorm1d(out_size))
                    self.FC.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(negative_slope=0.3))
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dropout_rate))

        #self.FC = nn.ModuleList(self.FC)

        print("\tMean/Var Layer in/out:", layer_sizes[-1], z_dimension)
        self.mean_encoder = nn.Linear(layer_sizes[-1], z_dimension)
        


    def forward(self, x: torch.Tensor):
        if self.FC is not None:
            x = self.FC(x)
        mean = self.mean_encoder(x)
        
        return mean


# =============================================================================
# PDecoderBio（Soft Mask 版本）
# 核心改动：从 Hard Mask 改为 Soft Mask
# 
# 【原始 Hard Mask 做法（已废弃）】：
#   self.sparse_linear.weight.data *= self.mask.T.to(z.device)  # 原地清零
#   self.sparse_linear.weight.data.clamp_(min=0)                # 原地非负
#   问题：.data 原地操作破坏 autograd 计算图，mask=0 的权重永远无法恢复
#
# 【新 Soft Mask 做法（expiMap 论文公式5-6）】：
#   1. 不再对权重做原地清零，所有权重都可以自由学习
#   2. 通过 L1 正则化惩罚非通路基因的权重，使其倾向于零但允许非零
#   3. 非负约束通过 torch.clamp（非原地）实现，保持计算图完整
#   4. 新增 get_soft_mask_loss(gamma) 方法供 Trainer 调用
# =============================================================================
class PDecoderBio(PDecoder):
    """
    Soft-mask 版本的 Perturb-decoder（参照 expiMap, Lotfollahi et al., Nat Cell Biol 2023）。
    
    与 Hard Mask 的区别：
    - Hard Mask：直接将非通路基因权重清零（weight.data *= mask），永远不可恢复
    - Soft Mask：对非通路基因权重施加 L1 稀疏正则化，允许学习但倾向于零
    
    公式参考 expiMap 论文：
    - 公式(5): R_γ(W) = γ * Σ_j ||W_{:,j} ⊙ M_{:,j}||₁
    - 公式(6): M_ij = 1 if B_ij=0 (非通路基因), M_ij = 0 otherwise
    """
    def __init__(self, z_dimension, layer_sizes, x_dimension, dropout_rate, mask=None):
        # 显式初始化，不调用父类的 __init__ 避免冲突
        nn.Module.__init__(self) 
        self.x_dim = x_dimension
        self.gmv_dim = 50 
        
        # =====================================================================
        # 【改动1】构建 Soft Mask 矩阵 M（expiMap 公式6）
        # 原始 mask B: [50, x_dim], B_ij=1 表示基因 j 属于通路 i
        # Soft Mask M: [50, x_dim], M_ij=1 表示非通路基因（需要被 L1 惩罚）
        # =====================================================================
        if mask is not None:
            soft_mask = 1.0 - mask.float()  # 反转：非通路基因位置为 1
            self.register_buffer('soft_mask', soft_mask)  # [50, x_dim]，自动跟随 device
        else:
            self.register_buffer('soft_mask', None)
        
        # 路径一：50维稀疏路径 (Pathway 效果)
        # 【改动2】权重不再做原地清零，改由 soft mask L1 正则化控制稀疏性
        self.sparse_linear = nn.Linear(self.gmv_dim, x_dimension, bias=False)
        
        # 路径二：全连接路径 (补偿性能，处理剩余维度)
        self.free_dim = z_dimension - self.gmv_dim
        self.res_decoder = nn.Sequential(
            nn.Linear(self.free_dim, layer_sizes[0]),
            nn.BatchNorm1d(layer_sizes[0]),
            nn.LeakyReLU(0.3),
            nn.Linear(layer_sizes[0], x_dimension * 2)
        )

    def forward(self, z):
        # 1. 拆分隐变量：前 50 维给生物通路，其余给自由节点
        z_bio = z[:, :self.gmv_dim]
        z_free = z[:, self.gmv_dim:]
        
        # =====================================================================
        # 【改动3】使用 autograd 兼容的非负约束（非原地操作）
        # 旧代码：
        #   self.sparse_linear.weight.data *= self.mask.T.to(z.device)  ← 原地清零（hard mask）
        #   self.sparse_linear.weight.data.clamp_(min=0)                ← 原地 clamp
        # 新代码：
        #   torch.clamp（非原地）+ F.linear（保持计算图）
        # =====================================================================
        clamped_weight = torch.clamp(self.sparse_linear.weight, min=0)  # [x_dim, 50]
        out_sparse = F.linear(z_bio, clamped_weight)  # 使用 functional 接口，保持计算图完整
        
        # 3. 执行补偿路径
        out_free = self.res_decoder(z_free)
        
        # 4. 特征融合
        dim = self.x_dim
        recon_means = out_sparse + out_free[:, :dim]
        recon_vars = out_free[:, dim:]
        
        # 拼接均值和方差返回
        return torch.cat((torch.relu(recon_means), recon_vars), dim=1)
    
    # =========================================================================
    # 【改动4】新增：Soft Mask L1 正则化损失函数
    # =========================================================================
    def get_soft_mask_loss(self, gamma=0.1):
        """
        计算 Soft Mask L1 正则化损失（expiMap 公式5）。
        
        R_γ(W) = γ * Σ_j ||W_{:,j} ⊙ M_{:,j}||₁
        
        仅对非通路基因的权重施加 L1 惩罚：
        - 通路内基因（M=0）：权重自由学习，不受惩罚
        - 非通路基因（M=1）：权重被 L1 惩罚，倾向于零但允许非零
        
        Parameters
        ----------
        gamma : float
            L1 正则化强度。越大越接近 hard mask，越小约束越松。
            推荐值：0.01, 0.05, 0.1, 0.5
        
        Returns
        -------
        soft_loss : torch.Tensor or float
            Soft mask L1 正则化损失，可直接加入总 loss
        """
        if self.soft_mask is None:
            return 0.0
        
        W = self.sparse_linear.weight  # [x_dim, 50]
        M = self.soft_mask.T           # [x_dim, 50]（转置以匹配权重维度）
        
        # 对非通路基因权重施加 L1 正则化
        soft_loss = gamma * torch.sum(torch.abs(W * M))
        return soft_loss


class PAdaptor(nn.Module):
    """
    Constructs the  Perturb-adaptor. This class implements the adaptor part of PRnet. It will chemical perturbation in to 'comb_num' latent space.

    """

    def __init__(self, layer_sizes: list, comb_dimension: int, dropout_rate: float):
        super().__init__() # to run nn.Module's init method

        # encoder architecture
        self.FC = None
        if len(layer_sizes) > 1:
            print("Encoder Architecture:")
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    print("\tInput Layer in, out:", in_size, out_size)
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
                else:
                    print("\tHidden Layer", i, "in/out:", in_size, out_size)
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                    self.FC.add_module("N{:d}".format(i), module=nn.BatchNorm1d(out_size))
                    self.FC.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(negative_slope=0.3))
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dropout_rate))

        print("\tComb Layer in/out:", layer_sizes[-1], comb_dimension)
        self.comb_encoder = nn.Linear(layer_sizes[-1], comb_dimension)


    def forward(self, x: torch.Tensor):
        if self.FC is not None:
            x = self.FC(x)
        comb_encode = self.comb_encoder(x)
        return comb_encode
