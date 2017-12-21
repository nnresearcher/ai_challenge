from keras.models import load_model 
from PIL import Image
import numpy as np
from keras.preprocessing.image import  img_to_array
infile = 'E:/spyder_workspace/ai_challenger/data/4.jpg'



img = Image.open(infile)
out =img.resize((227,227),Image.ANTIALIAS)
X = img_to_array(out)
model_name = 'ai_model.h5'
model = load_model(model_name) 
pre_test=model.predict(np.array([X]),batch_size=1)
print('pre_test:',pre_test)
a = pre_test[0][0]
m = 0
for i in range(len(pre_test[0])):
    if a<pre_test[0][i]:
        a = pre_test[0][i]
        m = i
print(a)
print(m)

