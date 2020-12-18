import torch
import torch.nn as nn
import os
import scipy.io as io
import numpy as np


class TextCNNDataFeature:
    def __init__(self, model, args):
        super(TextCNNDataFeature, self).__init__()

        self.model_dir = args.model_save_dir        # 模型存储路径
        self.feature_dir = args.feature_save_dir    # 特征存储路径

        model.load_state_dict(torch.load('./{}/best_steps.pt'.format(self.model_dir)))
        model.eval()
        model.dropout = nn.Identity()   # 随机丢失，设置为直接输出
        model.linear = nn.Identity()    # 线性映射，设置为直接输出

        self.model = model

    # 获取数据集特征
    def get_features_labels(self, data_iter):
        feature_list = []                                               # 数据类型为 List 列表
        label_list = []
        for batch in data_iter:
            with torch.no_grad():
                feature, label = batch.text.t_(), batch.label
            feature_tensor = self.model(feature)                        # 提取数据的特征，数据类型为 Tensor
            feature_numpy = feature_tensor.cuda().data.cpu().numpy()    # 转换数据格式，数据类型为 numpy 数组
            feature_list.append(feature_numpy)                          # 转换数据格式，数据类型为 List numpy 数组
            label_numpy = label.cuda().data.cpu().numpy()               # 标签特征
            label_list.append(label_numpy)

        features = np.array(feature_list)                               # 数据类型为 np.array
        features = np.reshape(features, (-1, features.shape[2]))        # 原向量：2×50×300，现在将最后一维固定，合并为100×300
        labels = np.array(label_list)
        labels = np.reshape(labels, (-1, 1))
        return features, labels

    # 保存特征和标签
    def save_features_labels(self, train_iter, val_iter, test_iter):
        f_train, f_tra_label = self.get_features_labels(train_iter)  # 获取训练特征
        f_val, f_val_label = self.get_features_labels(val_iter)      # 获取验证特征
        f_test, f_tes_label = self.get_features_labels(test_iter)    # 获取测试特征
        if not os.path.isdir(self.feature_dir):
            os.makedirs(self.feature_dir)
        save_name = 'feature.mat'
        save_path = os.path.join(self.feature_dir, save_name)
        io.savemat(save_path, {
            'T_tra_CNN': f_train, 'T_val_CNN': f_val, 'T_tes_CNN': f_test,
            'T_tra_CNN_label': f_tra_label, 'T_val_CNN_label': f_val_label, 'T_tes_CNN_label': f_tes_label
        })


