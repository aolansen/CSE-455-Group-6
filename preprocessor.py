import numpy as np
import pandas as pd



faces = pd.read_csv("fer2013.csv")
print(faces.head())

#Turn DataFram into List
pixels = faces["pixels"].tolist()
emotion = faces["emotion"].tolist()
Usage = faces["Usage"].tolist()

#Turning the list into an array 48X48 Function
def str_to_img(pixels):
        image =np.array(pixels.split(), dtype='float32')
        return image.reshape(48, 48)

#Going through the pixels and makes them into images
image_list=[]
for pixel in pixels:
    img = str_to_img(pixel)
    image_list.append(img)

#2 Dimensional Array
images = np.array(image_list)
#Making into 3 dimensional Array for use in CNN
images = np.expand_dims(images, -1) 