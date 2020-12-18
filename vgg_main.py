import torch
from PIL import Image
from init_seeds import Seeds
from vgg_model import VGG
import pandas as pd

if __name__ == '__main__':
    seeds = Seeds()
    seeds.init_seeds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('torch.cuda.is_available(): ', torch.cuda.is_available())

    vgg19 = VGG(device, 'vgg19')

    image_list = pd.read_csv('./list/train.csv')['image']
    for index, image_file in enumerate(image_list):
        input_image = Image.open('./dataset/' + image_file)
        feature_list = []
        feature_tensor = vgg19.get_image_feature(input_image)
        feature_numpy = feature_tensor.cuda().data.cpu().numpy()
        feature_list.append(feature_numpy)

