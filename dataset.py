import torch

import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchtext import data
from torchtext.data import get_tokenizer
from torch.nn import init





# VGG模型提取图像特征
class VGG:
    def __init__(self):
        super().__init__()

        # load image model
        image_model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        # or any of these variants
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13_bn', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)

        image_model.eval()  # 处于验证状态
        image_model.classifier._modules['6'] = nn.Identity()  # update the last layer
        self.image_model = image_model.to(DEVICE)

        # set image transform
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # 裁剪图片，输出为1×3×224×224
    def transform_image(self, image):
        input_tensor = self.preprocess(image)
        return input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # 获取图片特征，输出为4096维
    def feature_image(self, batch):
        batch = batch.to(DEVICE)
        with torch.no_grad():
            return self.image_model(batch)


# 获取图像的VGG特征
def get_image_tokenize(image_file):
    input_image = Image.open('./dataset/' + image_file)
    input_batch = vgg19.transform_image(input_image)
    return vgg19.feature_image(input_batch)


# 可用设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化VGG模型
vgg19 = VGG()
