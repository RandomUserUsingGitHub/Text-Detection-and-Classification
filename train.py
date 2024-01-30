import tensorflow as tf
import numpy as np
import time


from DataLoader import get_train_data
from Generator.Utils.FileUtils import get_model_path
from models import CustomModel, IoUCallback

batch_size = 16



def train_model():

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.set_visible_devices(physical_devices[0], 'GPU')

    [(train_images, train_labels, train_Bboxes),
     (validation_images, validation_labels, validation_Bboxes)] = get_train_data()

    instance = CustomModel()

    instance.model.summary()

    instance.model.compile(
        optimizer='adam',
        loss={'bbox_output': tf.keras.losses.Huber(delta=2.0), 'cls_output': 'binary_crossentropy'},
        metrics={'bbox_output': 'mae', 'cls_output': 'accuracy'}
        )

    IoU_callback = IoUCallback(validation_data=(np.array(validation_images), np.array(validation_Bboxes)), model=instance)


    start_time = time.time()
    instance.model.fit(
        np.array(train_images),
        {'bbox_output': np.array(train_Bboxes), 'cls_output': np.array(train_labels)},
        epochs=10,
        batch_size=batch_size,
        validation_data=(validation_images, {'bbox_output' : np.array(validation_Bboxes), 'cls_output' : np.array(validation_labels)}),
        callbacks=[IoU_callback]
    )
    end_time = time.time()

    print("Training time = ", end_time - start_time, " s")

    save_model = input("Save model? (y/n) ")
    if('y' in save_model.lower()):
        name = input("Enter model name: ")
        path = get_model_path(name)
        instance.model.save(path)

    