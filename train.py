import tensorflow as tf
import numpy as np
import time


from DataGenerator import DataGenerator
from Generator.Utils.FileUtils import get_model_path
from models import CustomModel, IoUCallback, IoUCallback2


batch_size = 16



def train_model():


    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.set_visible_devices(physical_devices[0], 'GPU')


    instance = CustomModel()

    instance.model.summary()

    instance.model.compile(
        optimizer='adam',
        loss={'bbox_output': 'mean_squared_error', 'cls_output': 'binary_crossentropy'},
        metrics={'bbox_output': 'mae', 'cls_output': 'accuracy'}
        )
    
    train_gen = DataGenerator("train", batch_size=20)
    val_gen = DataGenerator("validation", batch_size=20)



    # IoU_callback = IoUCallback(validation_data=(np.array(validation_images), np.array(validation_Bboxes)), model=instance)
    IoU_callback = IoUCallback2(generator=val_gen)



    start_time = time.time()
    instance.model.fit_generator(generator=train_gen,
                                validation_data=val_gen,
                                epochs=10,
                                use_multiprocessing=True,
                                workers=2,
                                callbacks=[IoU_callback],
                                verbose=1)
    end_time = time.time()

    print("Training time = ", end_time - start_time, " s")

    save_model = input("Save model? (y/n) ")
    if('y' in save_model.lower()):
        name = input("Enter model name: ")
        path = get_model_path(name)
        instance.model.save(path)

    