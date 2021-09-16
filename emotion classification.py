"""
@author: Melika_Golgoon
"""
import numpy as np
import pandas as pd
import os
import glob as gb
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#Path of Dataset
train_dir='/Users/Lenovo/Desktop/gh/emoyion-image/train'
test_dir='/Users/Lenovo/Desktop/gh/emoyion-image/test'
batch_S=64
#the number of images in each category
for folder in os.listdir(trrain_dir):
    files = gb.glob(pathname= str(train-dir +"/"+ folder + '/*.jpg'))
    print(f'for training data , found {len(files)} in folder {folder}')
for folder in os.listdir(test_dir):
    files = gb.glob(pathname= str(test-dir +"/"+ folder + '/*.jpg'))
    print(f'for training data , found {len(files)} in folder {folder}')
#see random images
import random
import matpotlib.pyplot as plt
import matpotlib.image as mpimg

def view_random_image(target_dir, target_class):
    target_folder = target_dirr + target_class
    raandom_image = random.sample(os.listdir(target_folder), 1)
    img = mpimp.imread(target_folder+'/'+random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis('off');
    print(f"image shape {img.shape}")
    return img

class_names = ['Anger','disgust','Fear','Hapiness','Sadness','Surprise']

plt.figure(figsize=(20, 10))
for i in range(18):
    plt.subplot(3, 6, i+1)
    class_name = random.choice(class_names)
    img = view_random_imagge(target_dir="/Users/Lenovo/Desktop/gh/emoyion-image/train",target_class=class_name)

#preparing data for training
from keras.preprocessing.image import imageDateGenerator

train_datagen = imageDateGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = imageDateGenerator(rescale = 1./255)
training-set = train_datagen.flow_from_directory(TRAIN_DIR, target_size = (128, 128), batch_S = BACH_SIZE, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(TEST_DIR, target_size = (128, 128), batch_S = BATCH_SIZE, class_mode = 'categorical')
#CNN
classifier = sequential()
#convolution
classifier.add(Conv2D(16, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
#Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
lassifier.add(MaxPooling2D(pool_size = (2, 2)))
#Flattening
classifier.add(Flatten())
#full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 6, activation = 'softmax'))
#compiling CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#model summary
classifier.summary()
istory = classifier.fit(training_set,
                         epochs = 50,
                         validation_data = test_set)
classifier.save('model1.h5')  #a HDF5 file
classifier.evaluate(test_set)
pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title('Loss')
plr.xlabel('epochs')
plt.ylabel('Loss')
pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title('Accuracy')
plr.xlabel('epochs')
plt.ylabel('Accuracy')
model_path = "model1.h5"loaded_model = keras.models.load_model(model_path)
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image myvideo_frames0_jpg.rf.8984dc0d89eac71f343dbedfddeefad7
image = cv2.imread("\Users\Lenovo\Desktop\gh\emotion-Image\test\Fear\Image myvideo_frames0_jpg.rf.8984dc0d89eac71f343dbedfddeefad7.jpg")
image_fromarray = Image.fromarray(image, 'RGB')
resize_image = image_fromarray.resize((128, 128))
expand_input = np.expand_dims(resize_image,axis=0)
input_data = np.array(expand_input)
input_data = input_data/255
pred = loaded_model.predict(input_data)
result = pred.argmax()
result
