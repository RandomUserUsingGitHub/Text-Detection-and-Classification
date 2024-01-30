import numpy as np
import pandas as pd

from PIL import Image
from Generator.Utils.FileUtils import define_csv_name, get_image_path

train_data_percentage = 0.8
validation_data_percentage = 0.1
test_data_percentage = 0.1

bbox_norm_value = 200

# train_images = images[:len_train]
# train_labels = labels[:len_train]
# train_Bboxes = bboxes[:len_train]

# validation_images = images[len_train:len_train+len_validation]
# validation_labels = labels[len_train:len_train+len_validation]
# validation_Bboxes = bboxes[len_train:len_train+len_validation]

test_images = []
test_labels = []
test_Bboxes = []


def load_data():
    df = pd.read_csv(define_csv_name())
    images_IDs = df["Image_ID"]
    labels = df["Class_Label"]
    Bboxes = df["Bbox"]
    images_in_numpy = []
    for ID in images_IDs:
        image = Image.open(get_image_path(ID))
        image = image.resize((200, 200))
        images_in_numpy.append(np.array(image))

    Bboxes_in_integer = [[int(integer) for integer in bbox.split(',')] for bbox in Bboxes]
    Bboxes_in_numpy = np.array(Bboxes_in_integer) / bbox_norm_value
    labels_in_numpy = labels.to_numpy()

    return images_in_numpy, Bboxes_in_numpy, labels_in_numpy

def get_train_data():
    images, bboxes, labels = load_data()

   

    len_dataset = int(len(labels))
    len_train = int(len_dataset * train_data_percentage)
    len_validation = int(len_dataset * validation_data_percentage)

    train_images = images[:len_train]
    train_labels = labels[:len_train]
    train_Bboxes = bboxes[:len_train]

    validation_images = images[len_train:len_train+len_validation]
    validation_labels = labels[len_train:len_train+len_validation]
    validation_Bboxes = bboxes[len_train:len_train+len_validation]

    train_images = np.array(train_images, dtype=float)
    validation_images = np.array(validation_images, dtype=float)

    train_images = train_images / 255
    validation_images = validation_images / 255


    return (train_images, train_labels, train_Bboxes), (validation_images, validation_labels, validation_Bboxes)


def get_test_data():
    images, bboxes, labels = load_data()
    len_dataset = int(len(labels))
    len_test = int(len_dataset * test_data_percentage)
    len_train = int(len_dataset * train_data_percentage)
    len_validation = int(len_dataset * validation_data_percentage)

    
    test_images = images[len_train+len_validation:len_dataset]
    test_labels = labels[len_train+len_validation:len_dataset]
    test_Bboxes = bboxes[len_train+len_validation:len_dataset]

    test_images = np.array(test_images, dtype=float)
    test_images = test_images / 255

    return (test_images, test_labels, test_Bboxes)

