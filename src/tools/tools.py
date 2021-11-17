import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,\
     Activation, Flatten, Conv2D, MaxPooling2D


class prepareData:

    def __init__(self, logger) -> None:
        self.logger = logger

        self.cwd = os.getcwd()
        self.data_dir = os.path.join(self.cwd, 'src', 'data')
        self.test_data_dir = os.path.join(self.cwd, 'src', 'test_data')
        self.categories = ['Basil', 'Chinar', 'Jatropha', 'Mango', 'Pongamia']

        self.img_size_x = 60
        self.img_size_y = 40

        self.show_example = False

        self.training_data = []
        self.train_X = []
        self.train_y = []

    def main(self):
        self.prepare_data()
        self.shuffle_data()
        self.neural_network_prep()
        self.normalize_data()
        self.cnn_model = self.create_CNN_model()
        self.predict()

    def prepare_data(self):
        self.logger.info("Preparing data...")
        for category in self.categories:
            data_path = os.path.join(self.data_dir, category)
            class_num = self.categories.index(category)
            if os.path.exists(data_path):
                for img in os.listdir(data_path):
                    img_path = os.path.join(data_path, img)
                    try:
                        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        resized_array = cv2.resize(
                            img_array,
                            (self.img_size_x, self.img_size_y))
                        self.training_data.append([resized_array, class_num])
                    except Exception:
                        self.logger.error(f"Broken image {img_path}")
            else:
                self.logger.warning("Incorrect path to data")

        if self.show_example:
            plt.imshow(resized_array, cmap="gray")
            plt.show()
            sys.exit(-1)

    def shuffle_data(self):
        self.logger.info("Shuffling data...")
        random.shuffle(self.training_data)

    def neural_network_prep(self):
        self.logger.info("Creating features and labels...")
        for features, label in self.training_data:
            self.train_X.append(features)
            self.train_y.append(label)

        self.train_X = np.array(
            self.train_X).reshape(-1, self.img_size_x, self.img_size_y, 1)
        self.train_y = np.array(self.train_y)

    def normalize_data(self):
        self.logger.info("Normalizing data...")
        self.train_X = self.train_X/255.0

    def create_CNN_model(self):
        self.logger.info("Creating the model...")
        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=self.train_X.shape[1:]))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))

        model.add(Dense(1))

        model.add(Activation('sigmoid'))

        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])

        model.fit(
            self.train_X, self.train_y,
            batch_size=32, validation_split=0.1,
            epochs=5)
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

                except Exception:
                    self.logger.error(f"Broken image {img_path}")
        else:
            self.logger.warning("Incorrect path to data")
