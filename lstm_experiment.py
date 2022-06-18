import copy
import os

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import logging as log
from torch.optim import AdamW
from back_tool import seed_everything, train_model, test_model, processing_data, MyDataset
from models import LstmNeuralNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
log.basicConfig(filename='log/lstm_log.txt', filemode='w', level=log.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

CFG = {  # 训练的参数配置
    'seed': 2,
    'epochs': 100,
    'batch_size': 16, # 注意：对于lstm 训练、测试、验证的batch_size必须一致
    'train_bs': 16,  # batch_size，可根据自己的显存调整
    'valid_bs': 16,
    'test_bs': 16,
    'lr': 1e-2,  # 学习率
    'num_workers': 0,
    'weight_decay': 2e-4,  # 权重衰减，防止过拟合
    'device': 0,
    'train_data': '模式识别数据集/train_data.pkl',
    'test_data': None,
    'val_data': '模式识别数据集/validation_data.pkl',
    'log': log,
    'train': True,  # 训练阶段为true 提交测试阶段为false
    'processing_data': False,
    'seq_length': 10,
    'sample_n': 100,  # 抽取的样本条数，None表示抽取全部样本
    'feature_count': 1,  # 使用的特征数目
    'lstm_hidden_size': 16,
    'lstm_num_layers': 2,
    'linear_hidden_size_1': 32,
    'linear_hidden_size_2': 8,
    'model_save_dir': './model_saved/lstm_model/',
}

seed_everything(CFG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if CFG['processing_data']:
    processing_data('模式识别数据集', 'train', n=CFG['sample_n'])
    processing_data('模式识别数据集', 'validation', n=CFG['sample_n'])
train_df = pd.read_pickle(CFG['train_data'])
test_df = None
val_df = pd.read_pickle(CFG['val_data'])
data_set = MyDataset(train_df, test_df, val_df, CFG)

model = LstmNeuralNet(CFG['feature_count'],
                      CFG['lstm_hidden_size'],
                      CFG['lstm_num_layers'],
                      CFG['linear_hidden_size_1'],
                      CFG['linear_hidden_size_2'],
                      data_set.get_label_num(),
                      CFG).to(device)

optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in tqdm(range(CFG['epochs'])):
        train_loss, train_acc = train_model(model, optimizer,  data_set, device,criterion, CFG)
        val_loss, val_acc = test_model(model, data_set,criterion, device, CFG)
        if val_acc > best_acc:
            if not os.path.exists(CFG['model_save_dir']):
                os.makedirs(CFG['model_save_dir'])
            torch.save(model.state_dict(), '{}{}_lstm_{}.pt'.format(CFG['model_save_dir'], epoch, round(val_acc, 4)))
            best_acc = val_acc