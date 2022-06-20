import pandas as pd
import  numpy as np
import os
import random
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.preprocessing import MinMaxScaler


def processing_data(compete_type_filename, data_filename, n=None):
    '''

    :param compete_type_filename: 竞赛类型 '模式识别数据集' or '模式识别数据集'
    :param data_filename:  数据集类型 'train' or 'validation'
    :param n: 导入的样本数 None全部导入
    :return:
    '''
    print('样本数:' , n)
    file_names = []
    for root, dirs, files in os.walk(f'./{compete_type_filename}/{data_filename}'):
        for name in files:
            file_names.append (os.path.join(root, name))

    X_seqs = []
    y_targets = []
    df_file_names = []

    # print(file_names)
    random.shuffle(file_names)
    for idx, file in enumerate(file_names):
        # print(file)
        X_seq_df = pd.read_csv(file)
        X_seq_df = X_seq_df.dropna(axis=1).drop(['ID'], axis=1)
        X_seq = X_seq_df.values
        y_targets.append(file.split('/')[-2])
        X_seq = X_seq.reshape(-1).tolist()
        X_seqs.append(X_seq)
        df_file_names.append(file)
        if n is not None and idx >= n: break
    res_df = pd.DataFrame({'X_seqs':X_seqs, 'y_targets':y_targets, 'file_names': df_file_names}, index=None)
    res_df.sort_values(by=['y_targets'], inplace=True, ignore_index=True)
    res_df.to_pickle(f"./{compete_type_filename}/{data_filename}_data.pkl")


# processing_data('模式识别数据集', 'train', n=None)
# processing_data('模式识别数据集', 'validation', n=None)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyDataset(Dataset):
    def __init__(self, train_df, test_df, val_df, CFG):
        self._label_to_idx = {}
        self._idx_to_label = {}
        labels = list(train_df['y_targets'].unique()) + list(val_df['y_targets'].unique())
        self.scaler = MinMaxScaler(feature_range=(0, 100))

        for label in labels:
            self.add_label(label)

        self.val_df = val_df
        self.test_df = test_df
        self.train_df = train_df
        self.CFG = CFG

        self._lookup_dict = {
            'train': self.train_df,
            'val': self.val_df,
            'test': self.test_df
        }

        self.set_data_type('train')

    def get_label_num(self):
        return len(self._label_to_idx)

    def get_df(self):
        return self._df

    def add_label(self, label):
        if label in self._label_to_idx:
            index = self._label_to_idx[label]
        else:
            index = len(self._label_to_idx)
            self._label_to_idx[label] = index
            self._idx_to_label[index] = label
        return index

    def lookup_label(self, label):
        return self._label_to_idx[label]

    def lookup_idx(self, idx):
        return self._idx_to_label[idx]

    def __len__(self):
        return len(self._df)

    def set_data_type(self, split = 'train'):
        self._split = split
        self._df = self._lookup_dict[split]


    def __getitem__(self, idx):
        label = self._df.y_targets.values[idx]
        features = self._df.X_seqs.values[idx]
        features = np.array(features).reshape(-1, 5)
        features = self.scaler.fit_transform(features)
        features = features[:self.CFG['seq_length'],-1] # 选取特征
        label_res = self.lookup_label(label)
        return features, label_res



class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def collate_fn(data):
    # TODO 对features进行处理
    return  data

def train_model(model,optimizer, data_set,device,criterion,CFG):  # 训练一个epoch
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    optimizer.zero_grad()
    data_set.set_data_type('train')

    train_loader = DataLoader(data_set, batch_size=CFG['train_bs'], shuffle=True,
                              num_workers=CFG['num_workers'])
    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    for step, (X, y) in enumerate(tk):
        X, y = X.to(device).to(torch.float32), y.to(device).long()
        # with autocast():  # 使用半精度训练
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(1) == y).sum().item() / y.size(0)

        losses.update(loss.item(), y.size(0))
        accs.update(acc, y.size(0))

        tk.set_postfix(loss=losses.avg, acc=accs.avg, stage='train')
        CFG['log'].info('train- loss:{}, acc:{}'.format(losses.avg, accs.avg))
    return losses.avg, accs.avg

def test_model(model, data_set, criterion, device, CFG):  # 验证
    model.eval()
    data_set.set_data_type('val')
    val_loader = DataLoader(data_set, batch_size=CFG['valid_bs'],  shuffle=False,
                            num_workers=CFG['num_workers'])
    losses = AverageMeter()
    accs = AverageMeter()
    y_truth, y_pred = [], []
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (X, y) in enumerate(tk):
            X, y = X.to(device).to(torch.float32), y.to(device).long()

            output = model(X)

            y_truth.extend(y.cpu().numpy())
            y_pred.extend(output.argmax(1).cpu().numpy())

            loss = criterion(output, y)

            acc = (output.argmax(1) == y).sum().item() / y.size(0)

            losses.update(loss.item(), y.size(0))
            accs.update(acc, y.size(0))

            tk.set_postfix(loss=losses.avg, acc=accs.avg, stage='val')
            CFG['log'].info('test - loss:{}, acc:{}'.format(losses.avg, accs.avg))
    # F1,GOLD,PRED,intersection,F11,GOLD1,PRED1,intersection1 = macro_f1(pred=y_pred,gold=y_truth)
    # print('GOLD:{} {}'.format(GOLD,GOLD1))
    # print('PRED:{} {}'.format(PRED,PRED1))
    # print('intersection:{} {}'.format(intersection,intersection1))
    # print('F1:{} {}'.format(F1,F11))
    return losses.avg, accs.avg

# # for test
# GLOB_CFG = { #训练的参数配置
#     'seed': 2,
#     'max_len': 100, #文本截断的最大长度
#     'epochs': 150,
#     'train_bs': 128, #batch_size，可根据自己的显存调整
#     'valid_bs': 256,
#     'test_bs': 256,
#     'lr': 8e-6, #学习率
#     'num_workers': 0,
#     'weight_decay': 2e-4, #权重衰减，防止过拟合
#     'device': 0,
#     'train_data': '模式识别数据集/train_data.pkl',
#     'test_data': None,
#     'val_data': '模式识别数据集/validation_data.pkl',
#     'train': True, #训练阶段为true 提交测试阶段为false
#     'processing_data': False,
#     'seq_length':300
# }
#
# train_df =  pd.read_pickle(GLOB_CFG['train_data'])
# test_df = None
# val_df = pd.read_pickle(GLOB_CFG['val_data'])
# data_set = MyDataset(train_df, test_df, val_df,GLOB_CFG)
# train_loader = DataLoader(data_set, batch_size=GLOB_CFG['train_bs'], shuffle=True,
#                               num_workers=GLOB_CFG['num_workers'])
#
# for X,y in train_loader:
#     print(X,y)
