import numpy as np
from keras.preprocessing import image
import os,re
# 列出目录下所有的图片
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.search(r'([\w]+\.(?:' + ext + '))', f)]
# 列出训练集猫的图片
files_dog = list_pictures('data/train/cats', 'jpg')

# ImageNet上预测为狗的类型
dogs = [
 'n02085620','n02085782','n02085936','n02086079'
,'n02086240','n02086646','n02086910','n02087046'
,'n02087394','n02088094','n02088238','n02088364'
,'n02088466','n02088632','n02089078','n02089867'
,'n02089973','n02090379','n02090622','n02090721'
,'n02091032','n02091134','n02091244','n02091467'
,'n02091635','n02091831','n02092002','n02092339'
,'n02093256','n02093428','n02093647','n02093754'
,'n02093859','n02093991','n02094114','n02094258'
,'n02094433','n02095314','n02095570','n02095889'
,'n02096051','n02096177','n02096294','n02096437'
,'n02096585','n02097047','n02097130','n02097209'
,'n02097298','n02097474','n02097658','n02098105'
,'n02098286','n02098413','n02099267','n02099429'
,'n02099601','n02099712','n02099849','n02100236'
,'n02100583','n02100735','n02100877','n02101006'
,'n02101388','n02101556','n02102040','n02102177'
,'n02102318','n02102480','n02102973','n02104029'
,'n02104365','n02105056','n02105162','n02105251'
,'n02105412','n02105505','n02105641','n02105855'
,'n02106030','n02106166','n02106382','n02106550'
,'n02106662','n02107142','n02107312','n02107574'
,'n02107683','n02107908','n02108000','n02108089'
,'n02108422','n02108551','n02108915','n02109047'
,'n02109525','n02109961','n02110063','n02110185'
,'n02110341','n02110627','n02110806','n02110958'
,'n02111129','n02111277','n02111500','n02111889'
,'n02112018','n02112137','n02112350','n02112706'
,'n02113023','n02113186','n02113624','n02113712'
,'n02113799','n02113978']

# ImageNet上预测为猫的类型
cats=[
'n02123045','n02123159','n02123394','n02123597'
,'n02124075','n02125311','n02127052']

from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions
import numpy as np
# 使用Imagenet上预训练的Xception进行预测
model = Xception(weights='imagenet')

# 记录出来文件的个数
num=0
# img_path = 'data/train/dogs/dog.1759.jpg'

# 遍历图片并进行预测，并且进行判断，若为预测的此图片不为狗，则打印出图片路径
for img_path in files_dog:
    num +=1
    print(num)
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features = model.predict(x)
    #     print('Predicted:', decode_predictions(features, top=5)[0])
    p = decode_predictions(features, top=50)[0]
    p1 = np.array(p)
    p2 = p1[ : ,0]
    for i in p2:
        flag = True
        if i in dogs:
            flag = False
            break
    if flag == True:
        print(img_path)
