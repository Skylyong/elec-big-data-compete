# 使用聚类分析和降维方法分析特征和标签的关系
# 时序数据聚类： https://codeantenna.com/a/GKsQnE4bWc

# http://liao.cpython.org/scipytutorial15.html

import numpy as np
from back_tool import processing_data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler

# 1. 设置配置文件

CFG = {  # 训练的参数配置
    'seed': 2,
    'train_data': '模式识别数据集/train_data.pkl',
    'test_data': None,
    'val_data': '模式识别数据集/validation_data.pkl',
    'processing_data': False,
    'seq_length': 500, # 序列长度
    'sample_n': 5000,  # 抽取的样本条数，None表示抽取全部样本
    'feature_count': [-1],  # 使用哪些列特征
    'scaler_type': 'max_min',
}
np.random.seed(CFG['seed'])

# 2. 导入数据集并处理数据
stand_scaler = StandardScaler()
max_min_scaler = MinMaxScaler(feature_range=(0, 100))


def df_to_matrix(df, label_encoder):
    y = label_encoder.transform(df.y_targets)
    X = np.zeros((df.X_seqs.values.shape[0], CFG['seq_length']*len(CFG['feature_count'])))
    for idx, x in  enumerate(df.X_seqs):
        x = np.array(x).reshape(-1, 5)
        x = x[:CFG['seq_length'],CFG['feature_count']]
        if CFG['scaler_type'] == 'max_min':
            x = max_min_scaler.fit_transform(x)
        else:
            x = stand_scaler.fit_transform(x)
        # x = max_min_scaler.fit_transform(x)
        x = x.reshape(-1)
        X[idx] = x
    return X, y

if CFG['processing_data']:
    processing_data('模式识别数据集', 'train', n=CFG['sample_n'])
    # processing_data('模式识别数据集', 'validation', n=CFG['sample_n'])
train_df = pd.read_pickle(CFG['train_data'])
test_df = None
# val_df = pd.read_pickle(CFG['val_data'])
label_encoder = LabelEncoder()
label_encoder.fit(train_df.y_targets)

#
X_train, y_train = df_to_matrix(train_df, label_encoder)
# X_val, y_val = df_to_matrix(val_df, label_encoder)
#
def plot_clusters(X, y):
    plt.scatter(X[:, 0], X[:, 1], s=5,c=y, marker='o', cmap=plt.cm.coolwarm)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


X, y = X_train, y_train
plot_clusters(X, y)
pca = PCA(n_components = 2)
new_X = pca.fit_transform(X)
plot_clusters(new_X, y)
print(pca.explained_variance_ratio_)

tsne = TSNE()
new_X = tsne.fit_transform(X)
plot_clusters(new_X, y)


import matplotlib.cm as cm
predict_result = {'kmeans': [], 'hierarchical_agglomerative':[], 'gaussian_mixture_model':[] }

start, end = 4, 5
for n_clusters in range(start, end ):
    clusterer = KMeans(n_clusters=n_clusters, random_state=5)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    predict_result['kmeans'].append({'hyp':{'n_clusters': n_clusters},\
                                     'cluster_labels': cluster_labels, \
                                     'silhouette_avg': silhouette_avg})
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
#

for n_clusters in range(start, end ):
    for linkage in ['ward', 'complete', 'average', 'single']:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage = linkage)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        predict_result['hierarchical_agglomerative'].append({'hyp':{'n_clusters':n_clusters, \
                                                             'linkage':linkage }, \
                                                             'cluster_labels':cluster_labels, \
                                                             'silhouette_avg':silhouette_avg})
        print("For n_clusters = %s and linkage = \'%s\' The average silhouette_score is :%s" % (n_clusters,linkage,silhouette_avg))


for n_clusters in range(start, end ):
    for covariance_type in ['full', 'tied', 'diag', 'spherical']:
        clusterer = GaussianMixture(n_components=n_clusters, covariance_type = covariance_type)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        predict_result['gaussian_mixture_model'].append({'hyp':{'n_clusters':n_clusters, \
                                                         'covariance_type':covariance_type},\
                                                         'cluster_labels':cluster_labels, \
                                                         'silhouette_avg':silhouette_avg})
        print("For n_clusters = %s and covariance_type = \'%s\' The average silhouette_score is :%s" % (n_clusters,covariance_type,silhouette_avg))
#
#
# # evaluation using ARI
print('Evaluation using ARI:')
for key, items in predict_result.items():
    max_score = 0
    best_hyp = items[0]['hyp']
    for item in items:
        score = adjusted_rand_score(y, item['cluster_labels'])
        if score > max_score:
            max_score = score
            best_hyp = item['hyp']
    print(key, end=': ')
    for key, item in best_hyp.items():
        print('%s: %s'%(key, item), end=' ')
    print('adjusted_rand_score:', max_score)
#
# # evaluation using NMI
print('\nEvaluation using NMI:')
for key, items in predict_result.items():
    max_score = 0
    best_hyp = items[0]['hyp']
    for item in items:
        score = normalized_mutual_info_score(y, item['cluster_labels'])
        if score > max_score:
            max_score = score
            best_hyp = item['hyp']
    print(key, end=': ')
    for key, item in best_hyp.items():
        print('%s: %s'%(key, item), end=' ')
    print('adjusted_rand_score:', max_score)
#
#
# pca = PCA(n_components = 2)
# new_X = pca.fit_transform(X)
# plot_clusters(new_X, y)
# print(pca.explained_variance_ratio_)
#
#
# tsne = TSNE()
# new_X = tsne.fit_transform(X)
# plot_clusters(new_X, y)
#
for key, items in predict_result.items():
    for item in items:
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(18, 7)

        ax1.set_title("The visualization of the clustered data with true label.")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.scatter(new_X[:, 0], new_X[:, 1], c=y, marker='o', cmap=plt.cm.coolwarm)

        ax2.scatter(new_X[:, 0], new_X[:, 1], c=item['cluster_labels'], marker='o', cmap=plt.cm.coolwarm)
        ax2.set_title("The visualization of the clustered data with predict label.")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        hyp = [key + '= '+ str(item) for key,item in item['hyp'].items()]
        hyp = ' and '.join(hyp)
        title = key + ' clustering on sample data with ' + hyp

        plt.suptitle((title), fontsize=14, fontweight='bold')
        plt.show()
