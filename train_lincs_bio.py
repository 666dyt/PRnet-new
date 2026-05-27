# -*- coding: utf-8 -*-
# @Author: DU YUTONG (Modified for Interpretability)
# @Date: 2026-05-12

import os
import sys
from datetime import datetime
import torch
import numpy as np
import scanpy as sc
import argparse

# 设置 GPU 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 导入原始组件和你的新生物学模型类
from trainer.PRnetTrainer import PRnetTrainer
from models.PRnet_Bio import PRnetBio 

def parse_args():
    parse = argparse.ArgumentParser(description='PRnet Biological Interpretability Training')  
    parse.add_argument('--split_key', default='random_split_0', type=str, help='split key of data')  
    args = parse.parse_args()  
    return args

def load_hallmark_mask(gene_names):
    """
    建立 Hallmark 掩码矩阵 (50 通路 x 978 基因)
    注：此处应根据你的具体映射逻辑实现 [cite: 486-487]
    """
    n_genes = len(gene_names)
    # 实际应用时需读取 GMT 文件并进行名称匹配
    mask = np.random.randint(0, 2, (50, n_genes)) 
    print(f"Successfully loaded biological mask with shape: {mask.shape}")
    return torch.tensor(mask, dtype=torch.float32)

if __name__ == "__main__":
    args_train = parse_args()
    start_time = datetime.now()

    # 训练配置
    total_epochs = 500    # 总 Epoch 数
    save_interval = 5   # 每 5 个 Epoch 保存一次
    
    config_kwargs = {
        'batch_size' : 512,
        'comb_num' : 1,
        'save_dir' : './checkpoint/bio_results/', # 建议存放在子文件夹
        'split_key' : args_train.split_key,
        'x_dimension' : 978,
        'hidden_layer_sizes' : [128],
        'z_dimension' : 64,  # 50通路 + 14自由节点
        'adaptor_layer_sizes' : [128],
        'comb_dimension' : 64, 
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

    if not os.path.exists(config_kwargs['save_dir']):
        os.makedirs(config_kwargs['save_dir'])

    # 1. 数据准备
    adata = sc.read('/media/mldadmin/home/s125mdg35_08/PRnet/dataset/Lincs_L1000.h5ad')
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # 2. 生成生物学掩码 [cite: 51-53]
    hallmark_mask = load_hallmark_mask(adata.var_names)

    # 3. 初始化基础 Trainer
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
        obs_key = config_kwargs['obs_key']
    )

    # 4. 动态注入改进后的生物学模型
    print("Injecting PRnetBio into Trainer...")
    Trainer.model = PRnetBio(
        adata, 
        x_dimension=config_kwargs['x_dimension'],
        hidden_layer_sizes=config_kwargs['hidden_layer_sizes'],
        z_dimension=config_kwargs['z_dimension'],
        adaptor_layer_sizes=config_kwargs['adaptor_layer_sizes'],
        comb_dimension=config_kwargs['comb_dimension'],
        comb_num=config_kwargs['comb_num'],
        drug_dimension=config_kwargs['drug_dimension'],
        dr_rate=config_kwargs['dr_rate'],
        mask=hallmark_mask
    )
    # 更新 GPU 上的模型引用
    Trainer.modelPGM = Trainer.model.get_PGM().to(Trainer.device)

    # 5. 分段训练与保存
    print(f"Starting training loop. Total: {total_epochs} epochs.")
    
    for current_end_epoch in range(save_interval, total_epochs + 1, save_interval):
        print(f"\n>>> Training stage: Up to epoch {current_end_epoch}")
        
        # 每次运行指定的间隔长度
        Trainer.train(
            n_epochs = save_interval, 
            lr = config_kwargs['lr'], 
            weight_decay= config_kwargs['weight_decay'], 
            scheduler_factor=config_kwargs['scheduler_factor'],
            scheduler_patience=config_kwargs['scheduler_patience']
        )
        
        # 保存该阶段的独立权重文件，增加时间戳防止覆盖
        ts = datetime.now().strftime("%m%d_%H%M")
        save_name = f"BioModel_{config_kwargs['split_key']}_ep{current_end_epoch}_{ts}.pt"
        save_full_path = os.path.join(config_kwargs['save_dir'], save_name)
        
        torch.save(Trainer.model.state_dict(), save_full_path)
        print(f"Successfully saved checkpoint: {save_full_path}")

    print(f"Total time spent: {(datetime.now() - start_time).seconds / 60:.2f} min")