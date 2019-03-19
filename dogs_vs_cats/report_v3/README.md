
# Dogs Vs Cats

kaggle竞赛地址：[dogs vs cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

参考：[ypwhs/dogs_vs_cats](https://github.com/ypwhs/dogs_vs_cats)

## 改变

1. 使用imagenet上的xception做异常值处理，并手动移除异常值图片。

2. 使用Xception预训练模型

3. 多增加全连接层，参数调节optimizer、epochs等

## 代码

1. 直接.py的文件，是上传到服务器运行的。

1. preprocess.ipynb 为train目录下创建cats和dogs目录，将猫和狗数据分别移到各自目录，test下创建test目录，并将测试图片移到此目录。

2. X_feature.py 为使用Xception模型提取特征。

3. Xception_pred_100.py 为使用Xception预测猫训练集中top100的概率中没有猫的图片（异常图片）进行打印到Xception_pred_100.log，其中有为了调试有记处理的图片个数，要看异常图片需使用命令：cat Xception_pred_100.log|grep cat。

4. Xception_pred.py 为使用Xception预测狗训练集中top50概率中没有狗的图片（异常图片）进行打印到Xception_pred_dog_50.log，日志中也有图片计数，查看命令：cat Xception_pred_dog_50.log|grep dog。

5. train_pred.ipynb 为加载提取的特征，为模型加上全连接层，训练预测。

6. Xception_pred_100_v1.py 和3类似，不同之处，此处使用的target_size=(299, 299)，3中使用的为（224, 224）。

7. report_v2.pdf 为报告。

# 结果
1. score: 0.05172


```python

```
