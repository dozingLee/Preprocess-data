# PascalSentenceDataset

This program is utility to download pascal sentence dataset.

## Installation

You can install by "git clone" command.

```
git clone https://github.com/rupy/PascalSentenceDataset.git
```

### Dependency

You must install some python libraries. Use pip command. Python>=2

```
PyQuery
```

## Usage

To download dataset, just run program as follow:

```
python pascal_sentence_dataset.py
```

You can also write code like this:

```python
# import
from pascal_sentence_dataset import PascalSentenceDataSet

# create instance
dataset = PascalSentenceDataSet()
# download images
dataset.download_images()
# download sentences
dataset.download_sentences()
# create correspondence data by dataset
# dataset.create_correspondence_data()

# create my pair data
dataset.create_pair_data()
# preprocess data
dataset.preprocess_data()
```

Return the following file list: (./list/)
- _correspondence.csv_ 1000 list data, titled: index, image
- _data_pairs.csv_ 1000 list data, titled: index, image, text, label
- _train.csv_ the training set with 800 image-text pairs (40 pairs per class)
- _validate.csv_ 100 the validation set with 100 image-text pairs (5 pairs per class)
- _test.csv_ 100 the testing set with 100 image-text pairs (5 pairs per class)



