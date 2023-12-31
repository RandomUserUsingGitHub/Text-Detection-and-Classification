import tensorflow as tf
import numpy as np

from DataLoader import get_train_data
from Generator.Utils.FileUtils import get_model_path
from models import CustomModel, IoUCallback



def train_model():
    [(train_images, train_labels, train_Bboxes),
     (validation_images, validation_labels, validation_Bboxes)] = get_train_data()

    
    instance = CustomModel(model_type="model2")

    instance.model.summary()

    instance.model.compile(
        optimizer='adam',  # Choose an optimizer (e.g., 'adam', 'sgd')
        loss={'bbox_output': 'mean_squared_error', 'cls_output': 'binary_crossentropy'},
        metrics={'bbox_output': 'mae', 'cls_output': 'accuracy'}
        )

    IoU_callback = IoUCallback(validation_data=(np.array(validation_images), np.array(validation_Bboxes)))

    # Train the model
    instance.model.fit(
        np.array(train_images),
        {'bbox_output': np.array(train_Bboxes), 'cls_output': np.array(train_labels)},
        epochs=20,
        batch_size=16,
        # validation_split=0.2,
        validation_data=(validation_images, {'bbox_output' : np.array(validation_Bboxes), 'cls_output' : np.array(validation_labels)}),
        callbacks=[IoU_callback]
    )

    save_model = input("Save model? (y/n) ")
    if('y' in save_model.lower()):
        name = input("Enter model name: ")
        path = get_model_path(name)
        instance.model.save(path)

    