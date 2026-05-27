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

# 3. 定义新的 PRnet
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


class PDecoderBio(PDecoder):
    def __init__(self, z_dimension, layer_sizes, x_dimension, dropout_rate, mask=None):
        # 显式初始化，不调用父类的 __init__ 避免冲突
        nn.Module.__init__(self) 
        self.x_dim = x_dimension
        self.gmv_dim = 50 
        self.mask = mask
        
        # 路径一：50维稀疏路径 (这就是你想要的 Pathway 效果)
        self.sparse_linear = nn.Linear(self.gmv_dim, x_dimension, bias=False)
        
        # 路径二：全连接路径 (补偿性能，处理剩余维度)
        # 假设总隐层 z_dimension 是 64 + c_dim + n_dim
        # 我们只取其中的自由节点部分进行全连接解码
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
        
        # 2. 执行生物路径：应用掩码并强制权重非负
        if self.mask is not None:
            self.sparse_linear.weight.data *= self.mask.T.to(z.device)
        self.sparse_linear.weight.data.clamp_(min=0)
        
        out_sparse = self.sparse_linear(z_bio)
        
        # 3. 执行补偿路径
        out_free = self.res_decoder(z_free)
        
        # 4. 特征融合
        dim = self.x_dim
        recon_means = out_sparse + out_free[:, :dim]
        recon_vars = out_free[:, dim:]
        
        # 拼接均值和方差返回
        return torch.cat((torch.relu(recon_means), recon_vars), dim=1)  


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
