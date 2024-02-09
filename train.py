import tensorflow as tf
import numpy as np
import time


from DataGenerator import DataGenerator
from Generator.Utils.FileUtils import get_model_path
from models import CustomModel, IoUCallback, IoUCallback2


batch_size = 40


def iou(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + 1e-7) / (union + 1e-7)



def train_model():


    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.set_visible_devices(physical_devices[0], 'GPU')


    instance = CustomModel()

    instance.model.summary()
    

    # tf.keras.optimizers.Adam()
    instance.model.compile(
        optimizer='adam',
        loss={'bbox_output': 'mean_squared_error', 'cls_output': 'binary_crossentropy'},
        metrics={'bbox_output': iou, 'cls_output': 'accuracy'}
        )
    
    train_gen = DataGenerator("train", batch_size=batch_size)
    val_gen = DataGenerator("validation", batch_size=batch_size)



    # IoU_callback = IoUCallback(validation_data=(np.array(validation_images), np.array(validation_Bboxes)), model=instance)
    IoU_callback = IoUCallback2(generator=val_gen)



    start_time = time.time()
    history =instance.model.fit_generator(generator=train_gen,
                                validation_data=val_gen,
                                epochs=15,
                                use_multiprocessing=False,
                                workers=1,
                                callbacks=[IoU_callback],
                                verbose=1)
    end_time = time.time()

    print("Training time = ", end_time - start_time, " s")

    save_model = input("Save model? (y/n) ")
    if('y' in save_model.lower()):
        name = input("Enter model name: ")
        path = get_model_path(name)
        instance.model.save(path)

    