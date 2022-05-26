import os
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(__file__)
image_dir = os.path.join(ROOT_DIR, 'dataset')

os.chdir(image_dir)

train_csv = pd.read_csv("train.csv")
train_csv['target'].value_counts()

X = train_csv['Image']
Y = train_csv['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=21, stratify=Y)

# print(Y_train.value_counts(),"\n")
# print(Y_test.value_counts(),"\n")

if not os.path.exists(os.path.join(ROOT_DIR, 'dataset', 'validation_dir')):
    os.mkdir('train_dir')
    for i in train_csv['target'].unique():
        os.mkdir('train_dir\\' + i)

    os.mkdir('validation_dir')
    for i in train_csv['target'].unique():
        os.mkdir('validation_dir\\' + i)

    for i in train_csv['target'].unique():
        for j in X_train[Y_train == i]:
            copyfile('train\\' + j, 'train_dir\\' + i + '\\' + j)

    for i in train_csv['target'].unique():
        for j in X_test[Y_test == i]:
            copyfile('train\\' + j, 'validation_dir\\' + i + '\\' + j)

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

image_size = 500
train_generator = train_datagen.flow_from_directory(
    os.path.join(ROOT_DIR, 'dataset', 'train_dir'),
    target_size=(image_size, image_size),
    batch_size=32,
    class_mode='sparse'
)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(ROOT_DIR, 'dataset', 'validation_dir'),
    target_size=(image_size, image_size),
    batch_size=32,
    class_mode='sparse'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(8, activation=tf.nn.softmax)
])


model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.Adam(), metrics=['accuracy'])


history = model.fit(train_generator,
                    epochs=11,
                    verbose=1,
                    validation_data=validation_generator
                    )

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(ROOT_DIR, 'dataset', 'test'),
    target_size=(image_size, image_size),
    color_mode='rgb',
    batch_size=1,
    class_mode='categorical',
    shuffle=False)

os.chdir(ROOT_DIR)
model.save('model.h5')



def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(image_size, image_size))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor

model = load_model("model.h5")

img_path = os.path.join(ROOT_DIR, 'dance.jpg')


new_image = load_image(img_path)

pred = model.predict(new_image)
print("predictions-")
print(pred)

list = {0:'baharatnatyam', 1:'kathak', 2:'kathakali',3:'kuchipudi',4:'manipuri',5:'mohiniyattam',6:'odissi',7:'sattriya'}
prediction = ""
max_val = .5
index = 0

for i in pred[0]:
    if(i > max_val):
        max_val = i
        prediction = list[index]
    index = index+1

print(prediction)
print(max_val, "%")
