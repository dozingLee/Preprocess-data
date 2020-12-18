from torchtext.data import get_tokenizer
from torchtext import data
from torch.nn import init


# 读取文本并分割成单词列表
def get_text_tokenize(text_file):
    with open('./sentence/' + text_file, 'r') as f:
        output = f.read()
    tokenizer_item = get_tokenizer("basic_english")
    return tokenizer_item(output)


class TextCNNDataProcessor:
    def __init__(self, device, args):
        super(TextCNNDataProcessor, self).__init__()

        self.device = device  # 可用设备

        self.batch_size = args.batch_size  # batch大小

        self.static = args.static  # 静态加载词汇表
        self.vocab_size = None  # 词汇表的长度
        self.embedding_dim = args.embedding_dim  # 词的维度
        self.label_num = args.label_num  # 标签个数

        self.text_field = None  # 词汇表
        self.label_field = None  # 标签表

    # 加载数据
    def load_data(self):

        # 定义数据字段
        """
            如果需要设置文本的长度，则设置fix_length,否则torchtext自动将文本长度处理为最大样本长度
            text = data.Field(sequential=True, tokenize=tokenizer, fix_length=args.max_len, stop_words=stop_words)
        """
        text_field = data.Field(sequential=True, tokenize=get_text_tokenize, lower=True)
        label_field = data.Field(sequential=False, use_vocab=False)

        # 读取数据并根据定义的字段加载数据
        train, val, test = data.TabularDataset.splits(
            path='./list', train='train.csv', validation='val.csv', test='test.csv', format='csv', skip_header=True,
            fields=[('', None), ('index', None), ('image', None), ('text', text_field), ('label', label_field)]
        )

        # 加载并构建静态词向量表
        if self.static:
            text_field.build_vocab(train, val, test, vectors='glove.6B.300d')
            text_field.vocab.vectors.unk_init = init.xavier_uniform
        # 构建动态词向量，训练自动调整（此处不完整，缺少用户设置的embedding_dim）
        else:
            text_field.build_vocab(train, val, test)

        # 构建标签的词汇表（可用于解决非数字标签的数据集）
        # 标签build后：对于无标签<unk>为0，有标签0和1在build之后分别为1和2
        # 后续使用标签：在后面训练模型时标签减1相对应
        label_field.build_vocab(train, val, test)

        # 构建batch大小的数据集
        '''
        torchtext.data.Iterators
            Source: https://pytorch.org/text/stable/data.html
            
            (0) 默认shuffle在train=True的情况下为True
            (1) 在使用sort_key时，必须设置 shuffle=False 且 sort=True
            (2) 这里sort_key根据文本的长度进行排序
                sort_key用于对示例进行排序的键，以便将具有相似长度的示例批量组合在一起并最小化填充
            (3) lambda可以理解为箭头函数，输入x返回x的text属性值的长度
            (4) 这里val验证集的batch_size则正好为验证集的长度，不用进行细分
            (5) 这里device用于指定创建变量的设备（默认为cpu），如果使用gpu需要指定设备字符串
            
            实验说明：虽然这里定义了Iterator，但是正在实现Iterator的调用是在模型的训练过程，同时也是每个batch的过程
        '''
        train_iter = data.Iterator(train, batch_size=self.batch_size, shuffle=False, sort=True,
                                   sort_key=lambda x: len(x.text), device=self.device)

        val_iter = data.Iterator(val, batch_size=self.batch_size, shuffle=False, sort=True,
                                 sort_key=lambda x: len(x.text), device=self.device)

        test_iter = data.Iterator(test, batch_size=self.batch_size, train=False, sort=False, device=self.device)

        self.vocab_size = len(text_field.vocab)  # 词向量表的长度
        self.embedding_dim = text_field.vocab.vectors.size()[-1]  # 词向量维度
        self.label_num = len(label_field.vocab)  # 标签个数，标签词汇表的长度

        self.text_field = text_field
        self.label_field = label_field

        return train_iter, val_iter, test_iter

    # 返回当前数据加载的参数：vocab_size、embedding_dim、label_num
    def get_args(self):
        return self.vocab_size, self.embedding_dim, self.label_num

    # 返回当前构建的词汇表
    def get_build_vocab(self):
        return self.text_field.vocab.vectors

    # 返回当前构建的标签表
    # def get_label_vocab(self):
    #     return self.label_field.vocab.vectors
