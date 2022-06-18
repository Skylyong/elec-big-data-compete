import copy
import os

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import logging as log
from torch.optim import AdamW
from back_tool import seed_everything,train_model,test_model,processing_data,MyDataset
from models import MlpNeuralNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
log.basicConfig(filename='log/mlp_log.txt', filemode='w', level=log.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

GLOB_CFG = { #训练的参数配置
    'seed': 2,
    'max_len': 100, #文本截断的最大长度
    'epochs': 20,
    'train_bs': 128, #batch_size，可根据自己的显存调整
    'valid_bs': 256,
    'test_bs': 256,
    'lr': 8e-6, #学习率
    'num_workers': 0,
    'weight_decay': 2e-4, #权重衰减，防止过拟合
    'device': 0,
    'train_data': '模式识别数据集/train_data.pkl',
    'test_data': None,
    'val_data': '模式识别数据集/validation_data.pkl',
    'log': log,
    'train': True, #训练阶段为true 提交测试阶段为false
    'processing_data': True,
    'seq_length':300,
    'sample_n': 100, # 抽取的样本条数，None表示抽取全部样本
    'model_save_dir': './model_saved/mlp_model/'
}

seed_everything(GLOB_CFG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if GLOB_CFG['processing_data']:
    processing_data('模式识别数据集', 'train', n=GLOB_CFG['sample_n'])
    processing_data('模式识别数据集', 'validation', n=GLOB_CFG['sample_n'])
train_df =  pd.read_pickle(GLOB_CFG['train_data'])
test_df = None
val_df = pd.read_pickle(GLOB_CFG['val_data'])
data_set = MyDataset(train_df, test_df, val_df, GLOB_CFG)



MLP_CFG = GLOB_CFG.copy()
log.basicConfig(filename='log/mlp_log.txt', filemode='w', level=log.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
MLP_CFG['log'] = log
MLP_CFG['hidden_dim'] = 512
MLP_CFG['feature_count'] = 1


model = MlpNeuralNet(MLP_CFG['seq_length']*MLP_CFG['feature_count'],\
    MLP_CFG['hidden_dim'],\
    data_set.get_label_num()).to(device)
optimizer = AdamW(model.parameters(), lr=MLP_CFG['lr'], weight_decay=MLP_CFG['weight_decay'])
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in tqdm(range(MLP_CFG['epochs'])):
        train_loss, train_acc = train_model(model,optimizer,  data_set,device,criterion, MLP_CFG)
        val_loss, val_acc = test_model(model, data_set,criterion, device,MLP_CFG)
        if val_acc > best_acc:
            if not os.path.exists(GLOB_CFG['model_save_dir']):
                os.makedirs(GLOB_CFG['model_save_dir'])
            torch.save(model.state_dict(), '{}{}_mlp_{}.pt'.format(GLOB_CFG['model_save_dir'], epoch, round(val_acc, 4)))
            best_acc = val_acc