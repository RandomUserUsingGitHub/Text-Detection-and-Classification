import tensorflow as tf
import numpy as np

from DataLoader import read_data
from Generator.Utils.FileUtils import get_model_path, list_models



def train_model():
    (train_images, train_labels, train_Bboxes), (test_images, test_labels, test_Bboxes) = read_data()
    print(train_Bboxes[0])

    train_images = np.array(train_images, dtype=float)
    test_images = np.array(test_images, dtype=float)
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    input_layer = tf.keras.Input(shape=(200, 200, 1))


    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200,200,1)),
    #     tf.keras.layers.MaxPooling2D((2,2)),
    #     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D((3,3)),


    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(units=512, activation='relu'),
    #     tf.keras.layers.Dense(units=125, activation='relu'),
    #     tf.keras.layers.Dense(units=1, activation='sigmoid')
    # ])

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool1)
    maxpool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    flatten = tf.keras.layers.Flatten()(maxpool2)

    # Bounding Box Regression Branch
    bbox_fc1 = tf.keras.layers.Dense(64, activation='relu')(flatten)
    bbox_fc2 = tf.keras.layers.Dense(32, activation='relu')(bbox_fc1)
    bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox_output')(bbox_fc2)  # 4 for (x, y, w, h)

    # Binary Classification Branch
    cls_fc1 = tf.keras.layers.Dense(64, activation='relu')(flatten)
    cls_fc2 = tf.keras.layers.Dense(32, activation='relu')(cls_fc1)
    cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(cls_fc2)

    model = tf.keras.Model(inputs=input_layer, outputs=[bbox_output, cls_output])


    model.summary()

    train = input("Train model? (y/n) ")
    if('y' in train.lower()):
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # model.fit(train_images, train_labels, epochs=5)
        model.compile(
        optimizer='adam',  # Choose an optimizer (e.g., 'adam', 'sgd')
        loss={'bbox_output': 'mean_squared_error', 'cls_output': 'binary_crossentropy'},
        metrics={'bbox_output': 'mae', 'cls_output': 'accuracy'}
        )

        # Train the model
        model.fit(
            train_images,
            {'bbox_output': train_Bboxes, 'cls_output': train_labels},
            epochs=10,  # Choose the number of training epochs
            batch_size=32,  # Choose the batch size
            validation_split=0.2  # Optional: Specify the validation split
        )

        save_model = input("Save model? (y/n) ")
        if('y' in save_model.lower()):
            name = input("Enter model name: ")
            path = get_model_path(name)
            model.save(path)

    files = []
    print("List of models: ")
    for i, file in enumerate(list_models()):
        print(f'{i+1}. {file}')
        files.append(file)
    inp = int(input("-> "))
    name = files[inp-1]
    path = get_model_path(name)
    model = tf.keras.models.load_model(path)

    # test_loss = model.evaluate(test_images, test_labels)
    reshaped_image = np.expand_dims(test_images[565], axis=0)
    bbox, prediction = model.predict(reshaped_image)
    if(prediction[0] >= 0.5):
        prediction = 1
    else:
        prediction = 0
    print("Real value = ", test_labels[565], " | Prediciton = ", prediction)
    print("Real value = ", test_Bboxes[565], " | Prediciton = ", bbox)


