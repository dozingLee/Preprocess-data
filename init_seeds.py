import random
import numpy as np
from torch.backends import cudnn
import torch


# 随机种子，默认为0
class Seeds:
    def __init__(self, seed=0):
        super(Seeds, self).__init__()

        self.seed = seed

    def init_seeds(self):
        torch.manual_seed(self.seed)  # set the seed for generating random numbers
        torch.cuda.manual_seed(self.seed)  # set the seed for generating random numbers for the current GPU
        torch.cuda.manual_seed_all(self.seed)  # set the seed for generating random numbers on all GPUs
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.seed != 0:  # cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率
            cudnn.deterministic = True  # 不过实际上这个设置对精度影响不大，仅仅是小数点后几位的差别
            cudnn.benchmark = False  # 如果不是对精度要求极高，其实不太建议修改，因为会使计算效率降低
