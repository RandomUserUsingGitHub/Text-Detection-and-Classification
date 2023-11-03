import tensorflow as tf
import numpy as np

from DataLoader import read_data


def train_model():
    (train_images, train_labels), (test_images, test_labels) = read_data()

    train_images = np.array(train_images, dtype=float)
    test_images = np.array(test_images, dtype=float)
    train_images = train_images / 255.0
    test_images = test_images / 255.0


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200,200,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((3,3)),


        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dense(units=125, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)

    test_loss = model.evaluate(test_images, test_labels)