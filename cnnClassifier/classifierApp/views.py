from django.shortcuts import render
from django.core.files.storage import FileSystemStorage #To store the file in the system

import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('models\Cnn_CatOrDog_Classifier.h5')


# Create your views here.

def index(request):
    context={'a':1} 
    return render(request,'index.html',context)

def classifyImage(request):
     
    fileObj = request.FILES['filePath']
    fileStorage = FileSystemStorage()
    filePathName = fileStorage.save(fileObj.name,fileObj)
    filePathName = fileStorage.url(filePathName)
    test_img = '.'+filePathName
    test_image = image.load_img(test_img, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    # training_set.class_indices
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print(prediction)
    context={'filePathName':filePathName,'prediction':prediction}
    return render(request,'index.html',context)