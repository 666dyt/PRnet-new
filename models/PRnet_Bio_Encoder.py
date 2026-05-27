from xmlrpc.client import Boolean
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F 
from torch.autograd import Variable

import anndata
from anndata import AnnData
from typing import Optional, Union

from scipy import sparse
import scanpy as sc

class PRnet(nn.Module):
    """Model for PRnet class. 
    """

    def __init__(self, adata: AnnData, x_dimension=5000, hidden_layer_sizes: list = [128, 64], z_dimension: int = 10,  adaptor_layer_sizes: list = [128], comb_dimension: int = 50, comb_num: int = 2, drug_dimension: int=1024, dr_rate: float = 0.05, mask=None):
        super().__init__()
        self.adata = adata

        self.x_dim_ = x_dimension    #adata.n_vars
        self.c_dim = comb_dimension
        self.n_dim = 10
        self.z_dim_ = z_dimension
        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.dr_rate_ = dr_rate
        self.adaptor_layer_sizes = adaptor_layer_sizes
        self.comb_adapt_dim = comb_num * drug_dimension
        
        self.PGM = PGM(self.x_dim_, self.c_dim, self.n_dim, self.hidden_layer_sizes_, self.z_dim_, self.adaptor_layer_sizes, self.comb_adapt_dim, self.dr_rate_, mask=mask)


        self.is_trained_ = False
        self.trainer = None


    def get_latent(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor):
        """
        Map chemical perturbation and unperturbed state in to the latent space. This function will feed data
        in Perturb-encoder and compute the latent space coordinates
        for each sample in data.
        """
        latent = self.PGM.get_latent(x, c, n)
        latent = latent.cpu().detach() # to cpu then detach from the comput.graph
        return np.array(latent)

    def get_per_latent(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor):
        """
        Map chemical perturbation in to the latent space. 
        """

        latent = self.PGM.get_per_latent(x, c, n)
        latent = latent.cpu().detach() # to cpu then detach from the comput.graph
        return np.array(latent)

    def get_PGM(self):
        return self.PGM



class PGM(nn.Module):
    """perturbation-conditioned generative model involves  three  key  components:  the  Perturb-adaptor,  the  Perturb-encoder,  and  the Perturb-decoder.

    """
    def __init__(self, x_dim: int, c_dim: int, n_dim: int, hidden_layer_sizes: list = [128,128], z_dimension: int = 64, adaptor_layer_sizes: list = [128], comb_adapt_dim: int = 1024, dr_rate: float = 0.05, mask=None, **kwargs):
        super().__init__()
        
        # 1. 【必须先做属性赋值】否则后面调用 self.xxx 会报错
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.n_dim = n_dim
        self.z_dim = z_dimension
        self.hidden_layer_sizes = hidden_layer_sizes  # <--- 解决报错的关键在这里
        self.dr_rate = dr_rate
        self.adaptor_layer_sizes = adaptor_layer_sizes
        self.comb_adapt_dim = comb_adapt_dim

        # 路径 A: 原有 PEncoder
        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.x_dim + self.c_dim)
        self.encoder = PEncoder(encoder_layer_sizes, self.z_dim, self.dr_rate)

        # 路径 B: 新增生物通路分支 (PathwayBranch)
        # 注意：这里需要传入生物学掩码 mask
        self.pathway_branch = PathwayBranch(self.x_dim + self.c_dim, self.z_dim, mask)

        # 老师要求的：融合层 (相加后再过 MLP 和 ReLU)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU()
        )

        # 解码器保持不变 [cite: 260-318]
        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.x_dim*2)
        self.decoder = PDecoder(self.z_dim + self.c_dim + self.n_dim, decoder_layer_sizes, self.x_dim, self.dr_rate)
        # 推荐写法：既保留了作者逻辑，又简洁安全
        adaptor_in_sizes = [self.comb_adapt_dim] + self.adaptor_layer_sizes
        self.CombAdaptor = PAdaptor(adaptor_in_sizes, self.c_dim, self.dr_rate)
     

    def get_latent(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor):
        
        """
        Map chemical perturbation and unperturbed state in to the latent space. This function will feed data
        in Perturb-encoder and compute the latent space coordinates
        for each sample in data.
        """
        
        c = self.CombAdaptor(c)
        noise = (x, c)
        input_ = torch.cat(noise, 1)
        latent = self.encoder(input_)

        return latent


    def get_per_latent(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor):
        """
        Map `data` in to the latent space. This function will feed data
        in encoder part of VAE and compute the latent space coordinates
        for each sample in data.
        """
        latent = self.CombAdaptor(c)
        return latent


    def forward(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor):
        c = self.CombAdaptor(c)
        input_ = torch.cat((x, c), 1)

        # 1. 左路：原有编码器路径
        z_dense = self.encoder(input_) 

        # 2. 右路：生物通路路径
        z_pathway = self.pathway_branch(input_)

        # 3. 融合：相加后过 MLP 和 ReLU
        z_combined = z_dense + z_pathway
        z = self.fusion_layer(z_combined)

        # 4. 后续解码过程保持不变 [cite: 177-183]
        z_c_ = torch.cat((z, c, n), 1)
        x_hat = self.decoder(z_c_)
        return x_hat
    
    

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




class PDecoder(nn.Module):
    """
            Constructs the  Perturb-dncoder.  Decodes data from latent space to data space. It will transform constructed latent space to the previous space of data with means and log variances of n_dimensions = n_vars.
        """
    def __init__(self, z_dimension: int, layer_sizes: list, x_dimension: int, dropout_rate: float):
        super().__init__()

        layer_sizes = [z_dimension] + layer_sizes
        # decoder architecture
        print("Decoder Architecture:")
        # Create first Decoder layer
        self.FirstL = nn.Sequential()
        print("\tFirst Layer in, out", layer_sizes[0], layer_sizes[1])
        self.FirstL.add_module(name="L0", module=nn.Linear(layer_sizes[0], layer_sizes[1], bias=False))
        self.FirstL.add_module("N0", module=nn.BatchNorm1d(layer_sizes[1]))
        self.FirstL.add_module(name="A0", module=nn.LeakyReLU(negative_slope=0.3))
        self.FirstL.add_module(name="D0", module=nn.Dropout(p=dropout_rate))

        # Create all Decoder hidden layers
        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                if i+3 < len(layer_sizes):
                    print("\tHidden Layer", i+1, "in/out:", in_size, out_size)
                    self.HiddenL.add_module(name="L{:d}".format(i+1), module=nn.Linear(in_size, out_size, bias=False))
                    self.HiddenL.add_module("N{:d}".format(i+1), module=nn.BatchNorm1d(out_size, affine=True))
                    self.HiddenL.add_module(name="A{:d}".format(i+1), module=nn.LeakyReLU(negative_slope=0.3))
                    self.HiddenL.add_module(name="D{:d}".format(i+1), module=nn.Dropout(p=dropout_rate))
        else:
            self.HiddenL = None

        # Create Output Layers
        print("\tOutput Layer in/out: ", layer_sizes[-2], layer_sizes[-1], "\n")
        self.recon_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.relu = nn.ReLU()


    def forward(self, z: torch.Tensor):
        dec_latent = self.FirstL(z)

        # Compute Hidden Output
        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent

        # Compute Decoder Output
        recon_x = self.recon_decoder(x)
        dim = recon_x.size(1) // 2
        recon_x = torch.cat((self.relu(recon_x[:, :dim]), recon_x[:, dim:]), dim=1)
        return recon_x




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
    
class PathwayBranch(nn.Module):
    def __init__(self, input_dim, z_dim, mask):
        super().__init__()
        
        # 【修改这里】：先创建一个普通的 tensor
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        # 【核心修复】：使用 register_buffer 将其注册为模型的缓冲变量
        # 这样当你调用 model.to(device) 时，它会自动跟着被移到 GPU 上
        self.register_buffer('mask', mask_tensor)
        
        self.mask = torch.tensor(mask, dtype=torch.float32) # mask 维度应为 [input_dim, z_dim]
        self.weight = nn.Parameter(torch.Tensor(z_dim, input_dim))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        # 应用掩码：只有在 mask 为 1 的地方权重才有效
        masked_weight = self.weight * self.mask.t() 
        # 老师要求：保持可解释性通常需要非负约束 [cite: 408-410]
        masked_weight = torch.clamp(masked_weight, min=0) 
        return F.linear(x, masked_weight)