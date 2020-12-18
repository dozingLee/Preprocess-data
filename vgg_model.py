import torch.nn as nn
from torchvision import transforms
import torch


# VGG模型提取图像特征
'''
    device: available device (cpu or cuda)
    model_type: vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
        (batch norm version are suffixed with _bn)
'''


class VGG:
    def __init__(self, device, model_type='vgg19'):
        super().__init__()

        self.device = device
        self.model_type = model_type

        # load image model
        model = torch.hub.load('pytorch/vision:v0.6.0', self.model_type, pretrained=True)
        # or any of these variants
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13_bn', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)

        model.eval()                                      # validate status
        model.classifier._modules['6'] = nn.Identity()    # update the last layer
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
