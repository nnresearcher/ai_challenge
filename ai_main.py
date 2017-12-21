import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
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
    data_path = r'E:/spyder_workspace/ai_challenger/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
    data = load(data_path)
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
def nameinfo(data_type):
    if data_type =='validation':
        data_path = r'E:/spyder_workspace/ai_challenger/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
        file_path = r'E:/spyder_workspace/ai_challenger_scene_validation_20170908/scene_validation_images_20170908'
        base = 'E:/spyder_workspace/ai_challenger/data/validation/'    
    if data_type =='train':
        data_path = r'E:/spyder_workspace/ai_challenger/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
        file_path = r'E:/spyder_workspace/ai_challenger_scene_train_20170904/scene_train_images_20170904'
        base = 'E:/spyder_workspace/ai_challenger/data/train/'    
    #if data_type =='test':
    #    data_path = r'E:/spyder_workspace/ai_challenger/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
    #    file_path = r'E:/spyder_workspace/ai_challenger_scene_validation_20170908/scene_validation_images_20170908'
    #    base = 'E:/spyder_workspace/ai_challenger/data/validation/'
    return data_path,file_path,base
data_type = 'train'
data_path,file_path,base = nameinfo(data_type)


train_directory_path = 'E:/spyder_workspace/ai_challenger/data/train'
validation_directory_path = 'E:/spyder_workspace/ai_challenger/data/validation'

# Global Constants
NB_CLASS=80
LEARNING_RATE=0.01
MOMENTUM=0.9
ALPHA=0.0001
BETA=0.75
GAMMA=0.1
DROPOUT=0.5
WEIGHT_DECAY=0.0005
LRN2D_NORM=False
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'


def conv2D_lrn2d(x,filters,kernel_size,strides=(1,1),padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=WEIGHT_DECAY):
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None

    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)


    return x


def create_model():
    if DATA_FORMAT=='channels_first':
        INP_SHAPE=(3,227,227)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=1
    elif DATA_FORMAT=='channels_last':
        INP_SHAPE=(227,227,3)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=3

    x=conv2D_lrn2d(img_input,64,(11,11),4,padding='valid')
    #x=MaxPooling2D(pool_size=(3,3),strides=2,padding='valid',data_format=DATA_FORMAT)(x)
    x=conv2D_lrn2d(x,64,(5,5),1,padding='same')
    x=conv2D_lrn2d(x,32,(3,3),1,padding='same',lrn2d_norm=False)
    x=Flatten()(x)
    x=Dense(1000,activation='relu',init='glorot_uniform')(x)
    #x=Dropout(DROPOUT)(x)
    x=Dense(500,activation='relu',init='glorot_uniform')(x)
    #x=Dropout(DROPOUT)(x)
    x=Dense(output_dim=NB_CLASS,activation='softmax',init='glorot_uniform')(x)
    return x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT


def check_print():
    # Create the Model
    x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT=create_model()

    # Create a Keras Model
    model=Model(input=img_input,output=[x])
    model.summary()
    model.compile(optimizer='Adam',loss='categorical_crossentropy')
    print ('Model Compiled')
    
    test_datagen = ImageDataGenerator( rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    train_datagen = ImageDataGenerator(
                                       rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
            train_directory_path,  # this is the target directory
            target_size=(227, 227),  # all images will be resized to 150x150
            batch_size=32,
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            validation_directory_path,
            target_size=(227, 227),
            batch_size=32,
            class_mode='categorical')
    model.fit_generator(
            train_generator,
            samples_per_epoch=2000,
            nb_epoch=50,
            validation_data=validation_generator,
            nb_val_samples=800)
    model.save('ai_model.h5',overwrite=True)
if __name__=='__main__':
    check_print() 
