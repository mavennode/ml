import os

import shutil

dirname = 'data'
filenames = os.listdir('data')
train_dir = 'data/train'
test_dir = 'data/test'

# 解压并创建目录
def mkdir(dirname):
    if os.path.exists(dirname):
        shutil.unpack_archive(dirname+'/train.zip', 'data')
        shutil.unpack_archive(dirname+'/test.zip', 'data')
        os.mkdir('data/train/cats')
        os.mkdir('data/train/dogs')
        os.mkdir('data/test/test')

# 移动训练集中不同类型图片到指定目录
def mv(train_dir, test_dir):
    train_filenames = os.listdir(train_dir)
    test_filenames = os.listdir(test_dir)
    train_dog = filter(lambda x: x[:3] == 'dog', train_filenames)
    train_cat = filter(lambda x: x[:3] == 'cat', train_filenames)
    test = filter(lambda x: x[-3:] == 'jpg', test_filenames)
    for i in train_dog:
        shutil.move(train_dir+'/'+i, 'data/train/dogs')

    for i in train_cat:
        shutil.move(train_dir+'/'+i, 'data/train/cats')

    for i in test:
        shutil.move(test_dir+'/'+i, 'data/test/test')


mkdir(dirname)
mv(train_dir, test_dir)


