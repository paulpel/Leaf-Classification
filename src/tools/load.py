import logging
from sys import path
from tensorflow import keras
import os
import cv2
import numpy as np
import tensorflow as tf


class LoadModel:

    def __init__(self) -> None:
        

        self.cwd = os.getcwd()
        self.data_dir = os.path.join(self.cwd, 'src', 'data', 'full_data')
        self.test_data_dir = os.path.join(self.cwd, 'src', 'test_data')

        self.img_size_x = 60
        self.img_size_y = 40

        self.categories = [
            'Alstonia Scholaris', 'Arjun', 'Basil', 'Chinar', 'Gauva', 'Jamun',
            'Jatropha', 'Lemon', 'Mango', 'Pomegranate', 'Pongamia Pinnata'
            ]

    def main(self):
        self.model_g, self.model_c  = self.load_model()

    def load_model(self):
        path1 = os.path.join(self.cwd, 'src', 'models', 'greyscale_model')
        path2 = os.path.join(self.cwd, 'src', 'models', 'colormodel')
        model1 = keras.models.load_model(path1)
        model2 = keras.models.load_model(path2)
        return model1, model2

    def prepdata(self, img_path):
        data = []
        img_path = img_path
        try:
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
            resized_array = cv2.resize(
                img_array,
                (self.img_size_x, self.img_size_y))
            data.append(resized_array)
        except Exception as err:
                logging.error(
                    err)

        image = tf.keras.preprocessing.image.load_img(img_path, target_size=(60,40))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        data = np.array(data).reshape(-1, self.img_size_x, self.img_size_y, 1)
        
        pred_g = self.model_g.predict(data)
        pred_c = self.model_c.predict(input_arr)
        species_g = np.argmax(pred_g, axis = 1)
        species_c = np.argmax(pred_c, axis = 1)
        species_g = self.categories[species_g[0]]
        species_c = self.categories[species_c[0]]

        return species_g, species_c
