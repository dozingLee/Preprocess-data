import os
import scipy.io as io
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch

'''
    device: available device (cpu or cuda)
    model_type: vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
        (batch norm version are suffixed with _bn)
'''


# VGG模型提取图像特征
class VGGModelFeature:
    def __init__(self, model_type, args):
        super(VGGModelFeature, self).__init__()

        self.device = args.device
        self.model_type = model_type

        # load image model
        model = torch.hub.load('pytorch/vision:v0.6.0', model_type, pretrained=True)
        # or any of these variants
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13_bn', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)

        model.eval()  # validate status
        model.classifier._modules['6'] = nn.Identity()  # update the last layer

        self.model = model.to(self.device)

        # set image transform
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # 输入为图片，裁剪图片为1×3×224×224，输出为4096维
    def get_image_feature(self, image):
        input_tensor = self.preprocess(image)   # preprocess
        one_batch = input_tensor.unsqueeze(0)   # create a mini-batch as expected by the model
        one_batch = one_batch.to(self.device)   # to device
        with torch.no_grad():
            return self.model(one_batch)


# TextCNN模型提取文本特征
class TextCNNModelFeature:
    def __init__(self, model, args):
        super(TextCNNModelFeature, self).__init__()

        self.device = args.device
        self.model_dir = args.model_save_dir  # 模型存储路径
        self.feature_dir = args.feature_save_dir  # 特征存储路径

        model.load_state_dict(torch.load('./{}/best_steps.pt'.format(self.model_dir)))
        model.eval()
        model.dropout = nn.Identity()  # 随机丢失，设置为直接输出
        model.linear = nn.Identity()  # 线性映射，设置为直接输出

        self.model = model.to(self.device)

    def get_text_feature(self, text_batch):
        return self.model(text_batch)


class GeneratePairsFeature:
    def __init__(self, image_model_type, text_model_type, args):
        super(GeneratePairsFeature, self).__init__()

        self.feature_save_dir = args.feature_save_dir

        # vgg 图像提取特征模型
        self.image_model = VGGModelFeature(image_model_type, args)

        # text_cnn 文本提取特征模型
        self.text_model = TextCNNModelFeature(text_model_type, args)

    # 获取数据集特征
    def get_features_labels(self, data_iter):
        image_list, text_list, label_list = [], [], []  # 数据类型为 List 列表
        index = 0
        for batch in data_iter:
            with torch.no_grad():
                image, text, label = batch.image, batch.text.t_(), batch.label

            for image_item in image:
                image_url = data_iter.dataset.fields['image'].vocab.itos[image_item]

                index += 1
                print('\r{} / {}: \t{} \t ==> image feature'.format(index, len(data_iter), image_url))

                # image feature
                image_file = Image.open('./dataset/{}'.format(image_url))
                image_tensor = self.image_model.get_image_feature(image_file)
                image_numpy = image_tensor.cuda().data.cpu().numpy()
                image_list.append(image_numpy)

            # text feature
            text_tensor = self.text_model.get_text_feature(text)  # 提取数据的特征，数据类型为 Tensor
            text_numpy = text_tensor.cuda().data.cpu().numpy()  # 转换数据格式，数据类型为 numpy 数组
            text_list.append(text_numpy)  # 转换数据格式，数据类型为 List numpy 数组

            # label
            label_numpy = label.cuda().data.cpu().numpy()  # 标签特征
            label_list.append(label_numpy)

        image_features = np.array(image_list)
        image_features = np.reshape(image_features, (-1, image_features.shape[2]))

        text_features = np.array(text_list)  # 数据类型为 np.array
        text_features = np.reshape(text_features, (-1, text_features.shape[2]))  # 原向量：2×50×300，现在将最后一维固定，合并为100×300

        labels = np.array(label_list)
        labels = np.reshape(labels, (-1, 1))

        return image_features, text_features, labels

    # 保存特征和标签
    def save_features_labels(self, train_iter, val_iter, test_iter):
        I_train, T_train, L_train = self.get_features_labels(train_iter)  # 获取训练特征
        I_val, T_val, L_val = self.get_features_labels(val_iter)  # 获取验证特征
        I_test, T_test, L_test = self.get_features_labels(test_iter)  # 获取测试特征
        if not os.path.isdir(self.feature_save_dir):
            os.makedirs(self.feature_save_dir)
        save_name = 'feature.mat'
        save_path = os.path.join(self.feature_save_dir, save_name)
        io.savemat(save_path, {
            'I_tra_CNN': I_train, 'T_tra_vgg': T_train, 'L_train': L_train,
            'I_val_CNN': I_test, 'T_val_vgg': T_val, 'L_val': L_val,
            'I_tes_CNN': I_val, 'T_tes_vgg': T_test, 'L_test': L_test,
        })