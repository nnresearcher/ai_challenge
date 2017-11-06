import os
import json
import shutil 
def load(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def use_os(file_name,file_path):
    
    
    new_file_path = os.path.join(file_path, file_name)
    new_file_path.replace('\\','/')
    return new_file_path


def json_to_three_list(file_path,data_path):
    data = load(data_path)
    all_image_id=[]
    all_label_id=[]
    all_image_url=[]
    for i in range(len(data)):
        all_image_id.append(data[i]['image_id'])
        all_label_id.append(data[i]['label_id'])
        all_image_url.append(data[i]['image_url'])
    
    for i in range(len(all_image_id)):
        new_file_path = use_os(all_image_id[i],file_path)
        all_image_id[i] = new_file_path

    return all_image_id,all_label_id,all_image_url

def creat_package(base,num):

    i = 0
    for j in range(num):
        file_name = base+str(i)
        if os.path.exists(file_name):
            i=i+1
        else:
            os.mkdir(file_name)
            i=i+1

#def copy_image_to_new_file(base,file_path,data_path):
#    all_image_id,all_label_id,all_image_url = json_to_three_list(file_path,data_path)
#   for i in range(len(all_label_id)):
#        file_path = all_image_id[i]
#        new_file_path = base + str(all_label_id[i])
#        shutil.copy(file_path,new_file_path)

def copy_image_to_new_file(base,file_path,data_path):
    all_image_id,all_label_id,all_image_url = json_to_three_list(file_path,data_path)
    for i in range(len(all_label_id)):
        if all_label_id[i] == 0:
            file_path = all_image_id[i]
            new_file_path = base + str(all_label_id[i])
            shutil.copy(file_path,new_file_path)

def main(data_type):
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
    creat_package(base,80)
    copy_image_to_new_file(base,file_path,data_path)
    

main('train')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
