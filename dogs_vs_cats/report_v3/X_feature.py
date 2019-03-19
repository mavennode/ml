from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
import h5py

input_tensor = Input(shape=(299, 299, 3))
x = Lambda(preprocess_input)(input_tensor)
# 使用Xception进行迁移学习
base_model = Xception(input_tensor=x, weights='imagenet', include_top=False, pooling='avg')

model = Model(inputs=base_model.input, outputs=base_model.output)

gen = ImageDataGenerator()

# 由于服务器内存比较大，所以将batch_size设置为4000，这个是从1000慢慢提升到4000，峰值时期可以使用内存70%以上，cpu可以46核
train_generator = gen.flow_from_directory("data/train", (299, 299), shuffle=False,
                                          batch_size=4000)
# 验证集
validation_generator = gen.flow_from_directory("data/validation", (299, 299), shuffle=False,
                                          batch_size=4000)
# 测试集
test_generator = gen.flow_from_directory("data/test", (299, 299), shuffle=False,
                                         batch_size=4000, class_mode=None)


train = model.predict_generator(train_generator)
validation = model.predict_generator(validation_generator)
test = model.predict_generator(test_generator)

with h5py.File("Xception_feature_v1.h5") as h:
    h.create_dataset("train", data=train)
    h.create_dataset("validation", data=validation)
    h.create_dataset("test", data=test)
    h.create_dataset("train_label", data=train_generator.classes)
    h.create_dataset("validation_label", data=validation_generator.classes)
