import torch
import argparse

from init_seeds import Seeds
from textcnn_model import TextCNN
from textcnn_data_processer import TextCNNDataProcessor
from textcnn_trainer import TextCNNTrainer
from textcnn_data_feature import TextCNNDataFeature

parser = argparse.ArgumentParser(description='TextCNN text classifier')

parser.add_argument('-lr', type=float, default=0.001, help='学习率')
parser.add_argument('-epoch', type=int, default=20, help='epoch，每一个epoch大小所有样本都训练了一次')
parser.add_argument('-batch-size', type=int, default=50, help='batch，每一个batch大小更新一次参数')

parser.add_argument('-filter-num', type=int, default=100, help='卷积核的个数')
parser.add_argument('-filter-sizes', type=str, default='3,4,5', help='不同卷积核大小')
parser.add_argument('-dropout', type=float, default=0.5, help='随机失活率')

parser.add_argument('-static', type=bool, default=True, help='是否使用预训练词向量')
parser.add_argument('-label-num', type=int, default=20, help='标签个数（可自动获取）')
parser.add_argument('-embedding-dim', type=int, default=300, help='词向量的维度（预训练词向量可自动获取）')
parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调，不需要微调设置为True')

# parser.add_argument('-cuda', type=bool, default=True, help='是否使用GPU')
parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=50, help='经过多少iteration对验证集进行测试')
parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
parser.add_argument('-model-save-dir', type=str, default='model_dir', help='存储训练模型位置')
parser.add_argument('-feature-save-dir', type=str, default='feature_dir', help='存储文本TextCNN特征')

args = parser.parse_args()


if __name__ == '__main__':
    seeds = Seeds()
    seeds.init_seeds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('torch.cuda.is_available(): ', torch.cuda.is_available())

    print('1.正在加载数据...')
    processor = TextCNNDataProcessor(device, args)
    train_iter, val_iter, test_iter = processor.load_data()
    args.vocab_size, args.embedding_dim, args.label_num = processor.get_args()
    args.vectors = processor.get_build_vocab()
    print('\n加载数据完成！\n')

    print('2.正在加载模型...')
    model = TextCNN(args).to(device)
    print('\n加载模型完成！\n')

    print('3.开始训练模型...')
    trainer = TextCNNTrainer(args)
    trainer.train(train_iter, val_iter, model)
    print('\n训练模型完成！\n')

    print('4.测试静态模型...')
    trainer.test(model, test_iter)
    print('\n测试模型完成！\n')

    print('5.获取特征...')
    feature_extractor = TextCNNDataFeature(model, args)
    feature_extractor.save_features_labels(train_iter, val_iter, test_iter)
    print('\n获取特征完成！\n')
