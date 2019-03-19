
# Dogs Vs Cats

kaggle竞赛地址：[dogs vs cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

参考：[ypwhs/dogs_vs_cats](https://github.com/ypwhs/dogs_vs_cats)

## 改变

1. 增加数据增强

2. 使用Xception预训练模型

3. 一些参数的修改，optimizer、epochs等

## 代码
1. preprocess.ipynb 为train目录下创建cats和dogs目录，将猫和狗数据分别移到各自目录，test下创建test目录，并将测试图片移到此目录。

2. feature_extract.ipynb 为从Xception模型中提取特征

3. train_predict.ipynb 为训练预测

# 结果
1. score: 0.04171

2. val_loss: 0.0316

3. val_acc: 0.9906
