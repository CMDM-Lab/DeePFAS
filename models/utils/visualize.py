"""
Define virtualizable functions.

Author: Heng Wang
Date: 1/24/2024
"""
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from tsnecuda import TSNE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from ..config.models import PathConfig

#from umap import UMAP

def acc_validrate_curve(  # noqa: WPS213
    train_acc: List[float],
    val_acc: List[float],
    config: PathConfig
):

    plt.switch_backend('Agg')
    plt.plot(train_acc, 'r', label='train acc')
    plt.plot(val_acc, 'b', label='valid acc')
    plt.grid(axis='y')
    plt.ylabel('Acc rate')
    plt.xlabel('epoch')
    plt.legend()

    figure_name = 'Smiles_Acc_Curve.jpg'

    if os.path.isdir(config.figure_path) is not True:
        os.makedirs(config.figure_path, exist_ok=True)

    plt.savefig(os.path.join(config.figure_path, figure_name))

def loss_acc_curve(
    train_losses: List[float],
    train_acc: List[float],
    name: str,
    config: PathConfig      
):
    plt.switch_backend('Agg')
    plt.plot(train_losses, 'r', label='train loss')
    plt.plot(train_acc, 'b', label='train acc')
    plt.ylabel('loss-acc rate')
    plt.xlabel('epoch')
    plt.legend()

    figure_name = f'{name}-losses.jpg'
    if os.path.isdir(config.figure_path) is not True:
        os.makedirs(config.figure_path, exist_ok=True)

    plt.savefig(os.path.join(config.figure_path, figure_name))

def smiles_curve(  # noqa: WPS213, WPS211
    train_losses: List[float],
    val_losses: List[float],
    train_acc: List[float],
    val_acc: List[float],
    config: PathConfig
):

    plt.switch_backend('Agg')

    plt.plot(train_losses, 'r', label='train loss')
    plt.plot(val_losses, 'b', label='valid loss')
    plt.plot(train_acc, 'r--.', label='train acc')
    plt.plot(val_acc, 'b--.', label='valid acc')
    plt.grid(axis='y')
    plt.ylabel('loss-acc rate')
    plt.xlabel('epoch')
    plt.legend()

    figure_name = 'smiles.jpg'

    if os.path.isdir(config.figure_path) is not True:
        os.makedirs(config.figure_path, exist_ok=True)

    plt.savefig(os.path.join(config.figure_path, figure_name))

# def plot_rectangle(x_list, y_list, x_label, y_label, title, percentage=False):
#     d = {
#         x_label: x_list,
#         y_label: y_list
#     }
#     select_df = pd.DataFrame(d)
#     plt.figure(figsize=(12, 5))
#     plt.rcParams['savefig.dpi'] = 100
#     plt.rcParams['figure.dpi'] = 100
#     ax = select_df[[x_label, y_label]].plot(x=x_label, kind='bar', color='#5887ff')
#     plt.title(title, fontsize='18')
#     plt.xlabel(x_label, fontsize='8')
#     plt.ylabel(y_label, 
#                fontsize='14', 
#                rotation=360,
#                horizontalalignment='right', 
#                verticalalignment='top')
#     plt.xticks(fontsize=10, rotation=360)
#     plt.yticks(fontsize=14)
#     plt.ylim([0, 1000])
#     x = select_df[x_label].tolist()
#     y = select_df[y_label].tolist()
#     l = [i for i in range(len(select_df))]
#     a = '%' if percentage else ''
#     for i, (_x, _y) in enumerate(zip(l, y)): 
#         plt.text(_x, _y, f'{str(y[i])[0:3]} {a}', ha='center', va='bottom', color='black', fontsize=8)
    
#     plt.show()

def plot_rectangle(x_list, y_list, x_label, y_label, title, save_path, percentage=False, ylim=None):
    d = {
        x_label: x_list,
        y_label: y_list
    }
    select_df = pd.DataFrame(d)
    
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams['figure.dpi'] = 100

    # 创建图表对象
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制柱状图
    select_df.plot(x=x_label, y=y_label, kind='bar', color='#5887ff', ax=ax)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14, rotation=0, labelpad=30)
    ax.xaxis.set_tick_params(labelsize=10, rotation=45)  # 旋转 x 轴标签
    ax.yaxis.set_tick_params(labelsize=14)
    
    # 设置 y 轴范围
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, max(y_list) * 1.1)  # 让 y 轴稍微大于最大值 10%

    # 在每个条形上方添加数值标签
    a = '%' if percentage else ''
    for i, (_x, _y) in enumerate(zip(select_df[x_label], select_df[y_label])):
        ax.text(i, _y, f'{_y}{a}', ha='center', va='bottom', color='black', fontsize=8, rotation=45)
    plt.savefig(save_path)
    # 显示图表
    plt.show()

def plot_tsne(x, labels=None, num_classes=5, n_components=2, perplexity=15, lr=10):
    x_embedded = TSNE(n_components=n_components, 
                      perplexity=perplexity, 
                      learning_rate=lr).fit_transform(x)
    if labels is None:
        labels = np.random.randint(0, num_classes, len(x)) 
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=x_embedded[:, 0], y=x_embedded[:, 1], hue=labels, palette=sns.color_palette('hsv', num_classes))
    plt.title('t-SNE 2D Projection')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    os.makedirs('./figure_saved/latent_space', exist_ok=True)
    plt.savefig('./figure_saved/latent_space/tsne_plot.png')
    plt.show()

def plot_pca(x, labels=None, num_classes=5, n_components=2):
    x_embedded = PCA(n_components=n_components).fit_transform(x)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=x_embedded[:, 0], y=x_embedded[:, 1], hue=labels, palette=sns.color_palette('hsv', num_classes))
    plt.title('PCA 2D Projection')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    os.makedirs('./figure_saved/latent_space', exist_ok=True)
    plt.savefig('./figure_saved/latent_space/pca_plot.png')
    plt.show()

#def plot_umap(x, labels=None, num_classes=5, n_components=2, min_dist=0.1, n_neighbors=15):
#    x_embedded = UMAP(n_neighbors=n_neighbors,
#                      n_components=n_components,
#                      min_dist=min_dist).fit_transform(x)
#    plt.figure(figsize=(10, 7))
#    sns.scatterplot(x=x_embedded[:, 0], y=x_embedded[:, 1], hue=labels, palette=sns.color_palette('hsv', num_classes))
#    plt.title('UMAP 2D Projection')
#    plt.xlabel('Dimension 1')
#    plt.ylabel('Dimension 2')
#    plt.grid(True)
#    os.makedirs('./figure_saved/latent_space', exist_ok=True)
#    plt.savefig('./figure_saved/latent_space/umap_plot.png')
#    plt.show()
