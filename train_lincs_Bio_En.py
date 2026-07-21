import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import sys
print(sys.path)

import numpy as np
import torch
import argparse 
from datetime import datetime
import scanpy as sc
from trainer.PRnetTrainer_Bio_En import PRnetTrainer




def load_biological_mask(gmt_file_path, adata, c_dim=64, z_dim=64):
    """
    根据下载的 GMT 文件和 Lincs 数据集生成生物学掩码矩阵
    """
    # 1. 获取你的 978 个 Lincs 基因的顺序
    gene_list = list(adata.var_names)
    x_dim = len(gene_list) # 应该是 978
    
    # 2. 初始化全 0 的 Mask 矩阵 (基因维度, 潜在空间维度) -> (978, 64)
    gene_mask = np.zeros((x_dim, z_dim))
    
    # 3. 读取下载的 .gmt 文件
    with open(gmt_file_path, 'r') as f:
        lines = f.readlines()
        
    # Hallmark 正好有 50 个通路
    for i, line in enumerate(lines):
        if i >= 50: break # 确保不超过 50 个通路，留出后 14 个作为黑盒残差节点
        
        parts = line.strip().split('\t')
        pathway_name = parts[0]
        pathway_genes = parts[2:] # 从第三列开始是具体的基因名
        
        # 4. 匹配基因：如果在978个基因里，就把对应位置设为 1
        match_count = 0
        for gene in pathway_genes:
            if gene in gene_list:
                gene_idx = gene_list.index(gene)
                gene_mask[gene_idx, i] = 1.0  # 第 i 列代表第 i 个通路
                match_count += 1
        
        print(f"通路 {i+1}: {pathway_name} -> 匹配到了 {match_count} 个基因")

    # 5. 补齐药物嵌入维度 (c_dim = 64)
    # 因为输入是 [x, c] 拼起来的，所以下方要再垫上 64 行
    # 为了保证右路（通路层）只看基因表达而不看药物输入，这个地方全设为 0
    drug_mask = np.zeros((c_dim, z_dim))
    
    # 6. 上下拼接得到最终的 (1042, 64) 矩阵
    final_mask_np = np.vstack([gene_mask, drug_mask])
    final_mask = torch.tensor(final_mask_np, dtype=torch.float32)
    
    return final_mask

def parse_args():
    parse = argparse.ArgumentParser(description='perturbation-conditioned generative model')  
    parse.add_argument('--split_key', default='random_split_0', type=str, help='split key of data')  
    args = parse.parse_args()  
    return args



if __name__ == "__main__":
    args_train = parse_args()
    start_time = datetime.now()



    config_kwargs = {
        'batch_size' : 512,
        'comb_num' : 1,
        'save_dir' : './checkpoint/PRnet_Bio_En/',
        'save_frequency': 10, # Add：设定每 10 个 epoch 保存一次 [cite: 49]
        'n_epochs' : 500,
        'split_key' : args_train.split_key,
        'x_dimension' : 978,
        'hidden_layer_sizes' : [128],
        'z_dimension' : 64,
        'adaptor_layer_sizes' : [128],
        'comb_dimension' : 64, 
        #'drug_dimension': 1031,
        'drug_dimension': 1024,
        'dr_rate' : 0.05,
        'lr' : 1e-3, 
        'weight_decay' : 1e-8,
        'scheduler_factor' : 0.5,
        'scheduler_patience' : 10,
        'n_genes' : 20,
        'loss' : ['GUSS'], 
        'obs_key' : 'cov_drug_name'
    }  


    


    print(os.getcwd())

    # 直接跨文件夹调用旧的数据库
    adata = sc.read('/media/mldadmin/home/s125mdg35_08/PRnet/dataset/Lincs_L1000.h5ad')
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    gmt_path = '/media/mldadmin/home/s125mdg35_08/PRnet baseline/PRnet/dataset/h.all.v2026.1.Hs.symbols.gmt'
    print("正在生成生物学通路 Mask...")
    # 生成 mask
    mask = load_biological_mask(gmt_path, adata)
    
    
    Trainer = PRnetTrainer(
                            adata,
                            batch_size=config_kwargs['batch_size'],
                            comb_num=config_kwargs['comb_num'],
                            split_key=config_kwargs['split_key'],
                            model_save_dir=config_kwargs['save_dir'],
                            x_dimension=config_kwargs['x_dimension'],
                            hidden_layer_sizes=config_kwargs['hidden_layer_sizes'],
                            z_dimension=config_kwargs['z_dimension'],
                            adaptor_layer_sizes=config_kwargs['adaptor_layer_sizes'],
                            comb_dimension=config_kwargs['comb_dimension'],
                            drug_dimension=config_kwargs['drug_dimension'],
                            dr_rate=config_kwargs['dr_rate'],
                            n_genes=config_kwargs['n_genes'],
                            loss = config_kwargs['loss'],
                            obs_key = config_kwargs['obs_key'], 
                            mask = mask, #将 mask 传入 [cite: 24]
                            save_frequency = config_kwargs['save_frequency'] # 传递保存频率
                                )

    Trainer.train(
        n_epochs = config_kwargs['n_epochs'],
        lr = config_kwargs['lr'], 
        weight_decay= config_kwargs['weight_decay'], 
        scheduler_factor=config_kwargs['scheduler_factor'],
        scheduler_patience=config_kwargs['scheduler_patience'])

    end_time = datetime.now()

    during_time = (end_time-start_time).seconds/60

    print(f'start time: {start_time} end_time: {end_time} time:{during_time} min')
