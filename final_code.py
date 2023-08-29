# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 07:26:46 2022

@author: LINGARAJ
"""

from tensorflow.keras.models import load_model
from skimage import io
from sklearn import cluster
from tensorflow.keras.preprocessing import image
from skimage.transform import rescale, resize, downscale_local_mean

import numpy as np
import tensorflow as tf
from knn_clustring import segmenation


path = 'test_6.jpg'
test_image = image.load_img(path, target_size=(200, 200))
test_image = image.img_to_array(test_image)

test_image = test_image / 255.0
test_image = np.expand_dims(test_image, axis=0)



savedModel = load_model('gfgModel.h5')

prediction = savedModel.predict(test_image)



v = prediction[0][0]    

clas = 0
if v<0.7:
    print("CAP")
    clas = 0
else:
    print("Covid")
    clas = 1
    
    
    
per = segmenation(path, clas)


print("Affected Region",per)
print("Stage present at :",end="")
if per <25 and per>+0:
    print("Initial stage")
elif per <50 and per>=25:
    print("Middle stage")
elif per <75 and per>=50:
    print("Severe stage")
elif per <=100 and per>=75:
    print("Death")
