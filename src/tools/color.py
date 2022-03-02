import numpy as np
import os
import tensorflow as tf
import json

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import precision_recall_fscore_support


data_dir = '/Users/home/Programming/Git/Image Processing/src/data/full_data'
img_height = 60
img_width = 40
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  labels='inferred',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  labels='inferred',
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
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
              metrics=['accuracy'])

model.summary()

epochs = 30
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
predictions = model.predict(val_ds)
classes = np.argmax(predictions, axis=1)
print(len(classes))
path = os.path.join(os.getcwd(), 'colormodel')

testData = tf.keras.preprocessing.image_dataset_from_directory(
    '/Users/home/Programming/Git/Image Processing/src/test_data',
    labels='inferred',
    label_mode='categorical',
    seed=324893,
    image_size=(img_height, img_width),
    batch_size=batch_size)


predictions = model.predict(testData)
classes = np.argmax(predictions, axis=1)
labels = np.array([])

for x, y in testData:
    labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

precision, recall, fbeta, support = precision_recall_fscore_support(testData, classes)

data = {
    'precision': precision.tolist(),
    'recall': recall.tolist(),
    'fbeta': fbeta.tolist(),
    'support': support.tolist()
}

with open('test.json', 'w') as jf:
    json.dump(data, jf, indent=4)
