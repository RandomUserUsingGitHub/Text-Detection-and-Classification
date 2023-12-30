import tensorflow as tf
import numpy as np

from DataLoader import read_data
from Generator.Utils.FileUtils import get_model_path, list_models
from models import CustomModel, IoUCallback



def train_model():
    [(train_images, train_labels, train_Bboxes),
     (validation_images, validation_labels, validation_Bboxes),
     (test_images, test_labels, test_Bboxes)] = read_data()
    
    print(type(validation_Bboxes[0]))

    train_images = np.array(train_images, dtype=float)
    validation_images = np.array(validation_images, dtype=float)
    test_images = np.array(test_images, dtype=float)

    train_images = train_images / 255.0
    validation_images = validation_images / 255.0
    test_images = test_images / 255.0


    
    instance = CustomModel(model_type="model2")

    instance.model.summary()

    train = input("Train model? (y/n) ")
    if('y' in train.lower()):
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # model.fit(train_images, train_labels, epochs=5)
        instance.model.compile(
        optimizer='adam',  # Choose an optimizer (e.g., 'adam', 'sgd')
        loss={'bbox_output': 'mean_squared_error', 'cls_output': 'binary_crossentropy'},
        metrics={'bbox_output': 'mae', 'cls_output': 'accuracy'}
        )

        
        # Train the model
        instance.model.fit(
            np.array(train_images),
            {'bbox_output': np.array(train_Bboxes), 'cls_output': np.array(train_labels)},
            epochs=20,
            batch_size=32,
            # validation_split=0.2,
            validation_data=(validation_images, {'bbox_output' : np.array(validation_Bboxes), 'cls_output' : np.array(validation_labels)}),
            callbacks=[IoUCallback()]
        )

        save_model = input("Save model? (y/n) ")
        if('y' in save_model.lower()):
            name = input("Enter model name: ")
            path = get_model_path(name)
            instance.model.save(path)

    files = []
    print("List of models: ")
    for i, file in enumerate(list_models()):
        print(f'{i+1}. {file}')
        files.append(file)
    inp = int(input("-> "))
    name = files[inp-1]
    path = get_model_path(name)
    model = tf.keras.models.load_model(path)

    model.evaluate(np.array(test_images), {'bbox_output': np.array(test_Bboxes), 'cls_output': np.array(test_labels)},
               callbacks=[IoUCallback(test_data=(np.array(test_images), np.array(test_Bboxes)))])

    # test_loss = model.evaluate(test_images, test_labels)
    reshaped_image = np.expand_dims(test_images[300], axis=0)
    bbox, prediction = model.predict(reshaped_image)
    if(prediction[0] >= 0.5):
        prediction = 1
    else:
        prediction = 0
    print("Real value = ", test_labels[300], " | Prediciton = ", prediction)
    print("Real value = ", test_Bboxes[300], " | Prediciton = ", bbox)


