from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

from keras import backend as K


def euclidian_distance(inputs):
    x, y = inputs
    eucli = K.sum(K.square(x-y), keepdims=True)
    eucli = K.sqrt(K.maximum(eucli, K.epsilon()))
    return eucli


model = tf.keras.models.load_model(
    "model.h5", custom_objects={"euclidian_distance": euclidian_distance})


def link_to_array(link):

    img = tf.keras.preprocessing.image.load_img(link)
    img = img.resize((28, 28))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.rgb_to_grayscale(img)
    img = np.float32(img) / 255.
    img = np.expand_dims(img, axis=(0))
    return img


class_index = ["sac", "oreiller", "pot"]
class_images = [link_to_array("sac.jpg"), link_to_array(
    "pillow.jpg"), link_to_array("pot.jpg")]


def make_prediction(class_index, class_images, to_pred):

    liste_distance = list()
    for img in range(len(class_index)):
        pred = model.predict([class_images[img], to_pred])
        liste_distance.append(pred[0][0])

    index = np.argmin(liste_distance)
    # print(liste_distance)
    return "The image is a : " + str(class_index[index])


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/api", methods=["POST"])
def predict():
    img = request.json["image"]
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    ## img.shape = (1,28,28,1)
    pred = make_prediction(class_index, class_images, img)
    return pred


if __name__ == "__main__":
    app.run(debug=False, port=8000, host="0.0.0.0")
