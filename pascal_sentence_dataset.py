from urllib.parse import urljoin
from pyquery import PyQuery
import os
import requests
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


class PascalSentenceDataSet:
    DATASET_DIR = 'dataset/'
    SENTENCE_DIR = 'sentence/'
    PASCAL_SENTENCE_DATASET_URL = 'http://vision.cs.uiuc.edu/pascal-sentences/'

    def __init__(self):
        self.url = PascalSentenceDataSet.PASCAL_SENTENCE_DATASET_URL

    def download_images(self):
        dom = PyQuery(self.url)
        for img in dom('img').items():
            img_src = img.attr['src']
            category, img_file_name = os.path.split(img_src)

            # make category directories
            output_dir = PascalSentenceDataSet.DATASET_DIR + category
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            # download image
            output = os.path.join(output_dir, img_file_name)
            print(output)
            if img_src.startswith('http'):
                img_url = img_src
            else:
                img_url = urljoin(self.url, img_src)
            if os.path.isfile(output):
                print("Already downloaded, Skipping: %s" % output)
                continue
            print("Downloading: %s" % output)
            with open(output, 'wb') as f:

                while True:
                    result = requests.get(img_url)
                    raw = result.content
                    if result.status_code == 200:
                        f.write(raw)
                        break
                    print("error occurred while fetching img")
                    print("retry...")

    def download_sentences(self):
        dom = PyQuery(self.url)
        # tbody disappears in pyquery DOM
        for tr in dom('body>table>tr').items():
            img_src = tr('img').attr['src']
            category, img_file_name = os.path.split(img_src)

            # make category directories
            output_dir = PascalSentenceDataSet.SENTENCE_DIR + category
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            # download sentences
            head, tail = os.path.splitext(img_file_name)
            sentence_file_name = head + "txt"
            output = os.path.join(output_dir, sentence_file_name)
            if os.path.isfile(output):
                print("Already downloaded, Skipping: %s" % output)
                continue
            print("Downloading: %s" % output)
            with open(output, 'w') as f:
                for td in tr('table tr td').items():
                    f.write(td.text() + "\n")

    def create_correspondence_data(self):
        dom = PyQuery(self.url)
        writer = csv.writer(open('list/correspondence.csv', 'wb'))
        for i, img in enumerate(dom('img').items()):
            img_src = img.attr['src']
            print("%d => %s" % (i + 1, img_src))
            writer.writerow([i + 1, img_src])

    # my create pair data
    def create_pair_data(self):
        dom = PyQuery(self.url)
        writer = csv.writer(open('list/data_pairs.csv', 'w', newline=''))   # newline 每次添加时不增加新空行
        writer.writerow(['index', 'image', 'text', 'label'])
        category = ''
        cat_count = 0
        for i, img in enumerate(dom('img').items()):    # enumerate() 可以迭代返回计数
            img_src = img.attr['src']
            print("%d => %s" % (i, img_src))
            current_cat = img_src.split('/')[0]         # 当前分类
            if category != current_cat:                 # 判断是否是同一个类别
                cat_count += 1
                category = current_cat
            txt = img_src.replace('.jpg', 'txt')        # 获取文本
            writer.writerow([i, img_src, txt, cat_count])

    # 预处理数据，参考Demo：move-review
    def preprocess_data(self):
        # 读取csv文件
        # 有标题的直接通过data_list['title']的标题获取列数据
        # 没有标题，可以通过data_list['0']的index获取列数据
        data_list = pd.read_csv('list/data_pairs.csv')

        # 将数据分成0.8的训练集和0.2的另一数据集，以label标签为分层依据
        train, temp_list = train_test_split(data_list, test_size=0.2, stratify=data_list['label'])

        # 将原数据的0.2（另一数据集），分成测试机和验证集，同样以label标签作为分层依据
        val, test = train_test_split(temp_list, test_size=0.5, stratify=temp_list['label'])

        # 将文件保存到本地
        train.to_csv('./list/train.csv')
        val.to_csv('./list/val.csv')
        test.to_csv('./list/test.csv')


if __name__ == "__main__":
    # create instance
    dataset = PascalSentenceDataSet()
    # download images
    dataset.download_images()
    # download sentences
    dataset.download_sentences()
    # create correspondence data by dataset
    dataset.create_correspondence_data()
    # create my pair data
    dataset.create_pair_data()
    # preprocess data
    dataset.preprocess_data()
