import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,\
     Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from keras import backend as K


class prepareData:

    def __init__(self, logger) -> None:
        self.logger = logger

        self.cwd = os.getcwd()
        self.data_dir = os.path.join(self.cwd, 'src', 'data', 'full_data')
        self.test_data_dir = os.path.join(self.cwd, 'src', 'test_data')
        self.model_dir = os.path.join(self.cwd, 'src', 'models')
        self.categories = [
            'Alstonia Scholaris', 'Arjun', 'Basil', 'Chinar', 'Gauva', 'Jamun',
            'Jatropha', 'Lemon', 'Mango', 'Pomegranate', 'Pongamia Pinnata'
            ]
        self.categories = [
            'Alstonia Scholaris', 'Pongamia Pinnata'
            ]

        self.img_size_x = 60
        self.img_size_y = 40

        self.show_example = False

        self.training_data = []
        self.train_X = []
        self.train_y = []

        self.test_X = []
        self.test_y = []

        # model
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'adam'

    def main(self):
        self.greyscale_model()
        
    def greyscale_model(self):
        self.prepare_data()
        self.shuffle_data()
        self.neural_network_prep()
        self.cnn_model = self.create_CNN2_model()

        predictions = self.cnn_model.predict(self.test_X)
        classes = np.argmax(predictions, axis = 1)
        precision, recall, fbeta, support = precision_recall_fscore_support(self.test_y, classes)

        filename = os.path.join(self.model_dir, 'color_model')

        self.cnn_model.save(filename)

        data = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'fbeta': fbeta.tolist(),
            'support': support.tolist()
        }

        with open(filename + '.json', 'w') as jf:
            json.dump(data, jf, indent=4)

    def prepare_data(self):
        self.logger.info("Preparing data...")
        for category in self.categories:
            data_path = os.path.join(self.data_dir, category)
            class_num = self.categories.index(category)
            if os.path.exists(data_path):
                for img in os.listdir(data_path):
                    if '.DS_Store' in img:
                        continue
                    img_path = os.path.join(data_path, img)
                    try:
                        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        resized_array = cv2.resize(
                            img_array,
                            (self.img_size_x, self.img_size_y))
                        self.training_data.append([resized_array, class_num])
                    except Exception:
                        self.logger.error(
                            f"Broken or none image file: {img_path}")
            else:
                self.logger.warning(f"Incorrect path to data. path:{data_path}")
                

        if self.show_example:
            plt.imshow(resized_array, cmap="gray")
            plt.show()
            sys.exit(-1)

    def shuffle_data(self):
        self.logger.info("Shuffling data...")
        random.shuffle(self.training_data)

    def neural_network_prep(self):
        self.logger.info("Creating features and labels...")
        train, test =  train_test_split(self.training_data, test_size=0.2, random_state=25)

        
        for features, label in train:
            self.train_X.append(features)
            self.train_y.append(label)

        self.train_X = np.array(
            self.train_X).reshape(-1, self.img_size_x, self.img_size_y, 1)
        self.train_y = np.array(self.train_y)

        for features, label in test:
            self.test_X.append(features)
            self.test_y.append(label)

        self.test_X = np.array(
            self.train_X).reshape(-1, self.img_size_x, self.img_size_y, 1)
        self.test_y = np.array(self.train_y)

    def create_CNN2_model(self):
        num_classes = len(self.categories)

        model = Sequential([
            layers.Rescaling(1./255, input_shape=self.train_X.shape[1:]),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])  

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'])
        
        self.history = model.fit(
            self.train_X, self.train_y,
            batch_size=32, validation_split=0.1,
            epochs=30)

        return model

    def predict(self):
        if os.path.exists(self.test_data_dir):
            for img in os.listdir(self.test_data_dir):
                img_path = os.path.join(self.test_data_dir, img)
                try:
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    resized_array = cv2.resize(
                        img_array,
                        (self.img_size_x, self.img_size_y))
                    resized_array = resized_array.reshape(
                        -1, self.img_size_x, self.img_size_y, 1)
                    prediction = self.cnn_model.predict(resized_array)
                    plant_prediction = self.categories[int(prediction[0][0])]
                    print(f'PREDICTION OF {img}: {plant_prediction}')

                except Exception as err:
                    self.logger.error(err)
                    
        else:
            self.logger.warning("Incorrect path to data")
