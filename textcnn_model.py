import torch.nn as nn
import torch.nn.functional as F
import torch


class TextCNN(nn.Module):
    # 初始化
    def __init__(self, args):
        super(TextCNN, self).__init__()

        self.args = args  # 所有参数（arguments）
        filter_num = args.filter_num  # 卷积核的个数
        filter_sizes = [int(fsz) for fsz in args.filter_sizes.split(',')]   # 卷积核数组，默认[3,4,5]

        vocab_size = args.vocab_size        # 词汇表长度（根据数据集构建和确定）
        embedding_dim = args.embedding_dim  # 词向量维度（根据具体使用的词向量的维度确定）
        label_num = args.label_num          # 标签个数

        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 词向量矩阵

        if args.static:  # 静态词向量（如果使用预训练，词向量则提前加载，当不需要微调时设置freeze为True）
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)
            # model.embedding.weight.data.copy_(args.vocab.vectors)

            # 卷积网络层
        self.conv = nn.ModuleList([nn.Conv2d(1, filter_num, (fsz, embedding_dim)) for fsz in filter_sizes])

        self.dropout = nn.Dropout(args.dropout)  # dropout 随机丢弃，默认0.5

        self.linear = nn.Linear(len(filter_sizes) * filter_num, label_num)  # 确定输出向量维度：len(filter_sizes) * filter_num

    # 前馈网络
    def forward(self, x):
        # 输入维度为（batch_size, max_len）
        # max_len可以通过torchtext设置或自动获取为训练样本的最大长度

        # 经过embedding,x的维度为 (batch_size, max_len, embedding_dim)
        x = self.embedding(x)

        # 经过view函数x的维度变为 (batch_size, input_channel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.args.embedding_dim)

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_channel, w=max_len, h=1)
        x = [F.relu(conv(x)) for conv in self.conv]

        # 经过最大池化层,维度变为(batch_size, out_channel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]

        # 将不同卷积核运算结果维度 (batch, out_channel, w=1, h=1) 展平为 (batch, out_channel * w * h)
        x = [x_item.view(x_item.size(0), -1) for x_item in x]

        # 将不同卷积核提取的特征组合起来,维度变为 (batch, sum:out_channel * w * h)
        x = torch.cat(x, 1)

        # dropout层，随机丢失部分值（设置为0）
        x = self.dropout(x)

        # 全连接层
        logistic = self.linear(x)

        return logistic
