import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from keras.preprocessing.image import ImageDataGenerator

def load(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def use_os(file_name):
    file_path = r'E:/spyder_workspace/ai_challenger_scene_train_20170904/scene_train_images_20170904'
    new_file_path = os.path.join(file_path, file_name)
    new_file_path.replace('\\','/')
    return new_file_path

def imgread(path):
    '''Read an image array from a path'''
    img = mpimg.imread(path)
    return img

def json_to_three_list():
    data = load(r'E:/spyder_workspace/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json')
    all_image_id=[]
    all_label_id=[]
    all_image_url=[]
    for i in range(len(data)):
        all_image_id.append(data[i]['image_id'])
        all_label_id.append(data[i]['label_id'])
        all_image_url.append(data[i]['image_url'])
    
    for i in range(len(all_image_id)):
        file_path = use_os(all_image_id[i])
        all_image_id[i] = file_path

    return all_image_id,all_label_id,all_image_url

all_image_id,all_label_id,all_image_url = json_to_three_list()

img = imgread(all_image_id[0])
plt.imshow(img)


datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=False, 
                             rotation_range=0,
                             width_shift_range = 0.2, 
                             height_shift_range = 0.2, 
                             horizontal_flip = True)
i=0
for batch in datagen.flow_from_directory(all_image_id):
    i=i+1
    if i >20:
       break
   print (batch)