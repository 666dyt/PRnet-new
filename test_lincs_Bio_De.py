# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import argparse 
from datetime import datetime
import scanpy as sc
import numpy as np
import torch

# ==========================================
# 【核心修改点 1】：导入修改过 Decoder 的 Trainer
# ==========================================

from trainer.PRnetTrainer_Bio_En import PRnetTrainer 

# 引入 Mask 生成函数 
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
    parse = argparse.ArgumentParser(description='Test modified Decoder PRnet')  
    parse.add_argument('--split_key', default='lincs_split', type=str, help='split key of data')  
    args = parse.parse_args()  
    return args

if __name__ == "__main__":
    args_train = parse_args()
    start_time = datetime.now()
    
    # ==========================================
    # 【核心修改点 2】：指定新模型的 Checkpoint 和 Results 路径
    # ==========================================
    config_kwargs = {
        'batch_size' : 512,
        'comb_num' : 1,
        'save_dir' : './checkpoint/PRnet_Bio_De/',       
        'results_dir' : './results/PRnet_Bio_De/',       
        'split_key' : 'random_split_0',
        'x_dimension' : 978,
        'hidden_layer_sizes' : [128],
        'z_dimension' : 64,
        'adaptor_layer_sizes' : [128],
        'comb_dimension' : 64, 
        'drug_dimension': 1024,
        'dr_rate' : 0.05,
        'n_genes' : 20,
        'loss' : ['GUSS'], 
        'obs_key' : 'cov_drug_name'
    }  

    print("Testing Stage: 正在读取数据集...")
    adata = sc.read('/media/mldadmin/home/s125mdg35_08/PRnet/dataset/Lincs_L1000.h5ad')
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    # 生成 Mask 
    gmt_path = '/media/mldadmin/home/s125mdg35_08/PRnet/dataset/h.all.v2026.1.Hs.symbols.gmt'
    print("Testing Stage: 正在生成生物学通路 Mask...")
    mask = load_biological_mask(gmt_path, adata)

    os.makedirs(config_kwargs['results_dir'], exist_ok=True) 
    
    # 初始化 Trainer
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
        mask=mask 
    )

    # ==========================================
    # 【核心修改点 3】：加载最强权重文件
    # ==========================================
    checkpoint_path = os.path.join(config_kwargs['save_dir'], 'random_split_0_best_epoch_all.pt')
    print(f"Testing Stage: 正在加载模型权重进行测试: {checkpoint_path}")
    
    Trainer.test(checkpoint_path)

    end_time = datetime.now()
    during_time = (end_time-start_time).seconds/60
    print(f'Test Finished! start time: {start_time} end_time: {end_time} time:{during_time:.2f} min')
