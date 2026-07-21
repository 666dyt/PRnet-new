import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
print(sys.path)

import argparse 
from datetime import datetime
import scanpy as sc
import numpy as np
import torch

# 【修改点 1】：导入新的带有生物学分支的 Trainer
from trainer.PRnetTrainer_Bio_En import PRnetTrainer

# 【修改点 2】：引入 Mask 生成函数
def load_biological_mask(gmt_file_path, adata, c_dim=64, z_dim=64):
    gene_list = list(adata.var_names)
    x_dim = len(gene_list)
    gene_mask = np.zeros((x_dim, z_dim))
    
    with open(gmt_file_path, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if i >= 50: break 
        parts = line.strip().split('\t')
        pathway_genes = parts[2:] 
        for gene in pathway_genes:
            if gene in gene_list:
                gene_idx = gene_list.index(gene)
                gene_mask[gene_idx, i] = 1.0  
                
    drug_mask = np.zeros((c_dim, z_dim))
    final_mask_np = np.vstack([gene_mask, drug_mask])
    final_mask = torch.tensor(final_mask_np, dtype=torch.float32)
    return final_mask


def parse_args():
    parse = argparse.ArgumentParser(description='perturbation-conditioned generative model ')  
    parse.add_argument('--split_key', default='lincs_split', type=str, help='split key of data')  
    args = parse.parse_args()  
    return args


if __name__ == "__main__":
    args_train = parse_args()
    start_time = datetime.now()
    
    config_kwargs = {
        'batch_size' : 512,
        'comb_num' : 1,
        'save_dir' : './checkpoint/PRnet_Bio_En/',       
        'results_dir' : './results/PRnet_Bio_En/',      
        'n_epochs' : 100,
        'split_key' : 'random_split_0',
        'x_dimension' : 978,
        'hidden_layer_sizes' : [128],
        'z_dimension' : 64,
        'adaptor_layer_sizes' : [128],
        'comb_dimension' : 64, 
        'drug_dimension': 1024,
        'dr_rate' : 0.05,
        'lr' : 1e-3, 
        'weight_decay' : 1e-8,
        'scheduler_factor' : 0.5,
        'scheduler_patience' : 5,
        'n_genes' : 20,
        'loss' : ['GUSS'], 
        'obs_key' : 'cov_drug_name'
    }  

    print(os.getcwd())
    
    # 1. 读取数据
    adata = sc.read('/media/mldadmin/home/s125mdg35_08/PRnet/dataset/Lincs_L1000.h5ad')
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    # 2. 【修改点 4】：生成 Mask
    gmt_path = '/media/mldadmin/home/s125mdg35_08/PRnet baseline/PRnet/dataset/h.all.v2026.1.Hs.symbols.gmt'
    print("Testing Stage: 正在生成生物学通路 Mask...")
    mask = load_biological_mask(gmt_path, adata)

    # 3. 初始化 Trainer 并传入 mask
    os.makedirs(config_kwargs['results_dir'], exist_ok=True) # 自动创建结果文件夹
    
    Trainer = PRnetTrainer(
        adata,
        batch_size=config_kwargs['batch_size'],
        comb_num=config_kwargs['comb_num'],
        split_key=config_kwargs['split_key'],
        model_save_dir=config_kwargs['save_dir'],
        results_save_dir=config_kwargs['results_dir'],
        x_dimension=config_kwargs['x_dimension'],
        hidden_layer_sizes=config_kwargs['hidden_layer_sizes'],
        z_dimension=config_kwargs['z_dimension'],
        adaptor_layer_sizes=config_kwargs['adaptor_layer_sizes'],
        comb_dimension=config_kwargs['comb_dimension'],
        drug_dimension=config_kwargs['drug_dimension'],
        dr_rate=config_kwargs['dr_rate'],
        n_genes=config_kwargs['n_genes'],
        loss=config_kwargs['loss'],
        obs_key=config_kwargs['obs_key'],
        mask=mask  # 【修改点 5】：传入测试所需的 mask
    )

    # 【修改点 6】：调用底层的 test 方法
    checkpoint_path = './checkpoint/PRnet_Bio_En/random_split_0_best_epoch_all.pt'
    print(f"正在加载模型权重进行测试: {checkpoint_path}")
    Trainer.test(checkpoint_path)

    end_time = datetime.now()
    during_time = (end_time-start_time).seconds/60
    print(f'start time: {start_time} end_time: {end_time} time:{during_time} min')
