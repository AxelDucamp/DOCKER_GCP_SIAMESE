import requests
import numpy as np
import json
import tensorflow as tf

url = 'http://172.20.10.5:8000/api'


def link_to_array(link):

    img = tf.keras.preprocessing.image.load_img(link)
    img = img.resize((28, 28))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.rgb_to_grayscale(img)
    img = np.float32(img) / 255.
    # img = np.expand_dims(img, axis=(0))
    return img


image = link_to_array("p.jpg").tolist()

# print(image.shape)

myobj = {'image': image}

x = requests.post(url, json=myobj)

print(x.text)
