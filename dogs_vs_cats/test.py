# import pandas as pd
# from keras.preprocessing.image import *
# image_size = (224, 224)
# gen = ImageDataGenerator()
# test_generator = gen.flow_from_directory("/Users/matrix/notebook/udacity/dogs_vs_cats/data/test", image_size, shuffle=False,
#                                          batch_size=16, class_mode=None)
# print(enumerate(test_generator.filenames))
# for i, fname in enumerate(test_generator.filenames):
#     print(i, fname)
#     index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
#     df.set_value(index-1, 'label', y_pred[i])


from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import h5py
def write_gap(MODEL, modelName, image_size,lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    print(model)

write_gap(Xception, 'Xception', (299, 299), xception.preprocess_input)