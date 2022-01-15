import logging
from tensorflow import keras
import os
import cv2
import numpy as np


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
        self.model = self.load_model()

    def load_model(self):
        path = os.path.join(self.cwd, 'src', 'models', 'greyscale_model')
        model = keras.models.load_model(path)
        return model

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

        data = np.array(data).reshape(-1, self.img_size_x, self.img_size_y, 1)

        pred = self.model.predict(data)
        species = np.argmax(pred, axis = 1)
        species = self.categories[species[0]]
        return species
