import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Function to load and preprocess data
def load_and_preprocess_data(data_dir, target_size=(224, 224)):
    images = []
    labels = []

    team_labels = {
        'AlphaTauri F1 car': 0,
        'Ferrari F1 car': 1,
        'McLaren F1 car': 2,
        'Mercedes F1 car': 3,
        'Racing Point F1 car': 4,
        'Red Bull Racing F1 car': 5,
        'Renault F1 car': 6,
        'Williams F1 car': 7,

    }

    for sub_dir in team_labels.keys():
        path = os.path.join(data_dir, sub_dir)
        label = team_labels[sub_dir]

        for image_file in os.listdir(path):
            image_path = os.path.join(path, image_file)

   
            image = cv2.imread(image_path)

        
            if image is None:
                print(f"Error: Unable to read the image at {image_path}")
            else:
               
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, target_size)
                image = image / 255.0

                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)


data_dir = '/content/formula-one-cars/Formula One Cars'


images, labels = load_and_preprocess_data(data_dir)


train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(8, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) / 32, epochs=30, validation_data=(val_images, val_labels),
                    callbacks=[early_stopping])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
