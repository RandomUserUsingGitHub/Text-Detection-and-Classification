import numpy as np
import pandas as pd

from PIL import Image
from Generator.Utils.FileUtils import define_csv_name, get_image_path

tain_data_percentage = 0.8

def read_data():
    df = pd.read_csv(define_csv_name())
    images_IDs = df["Image_ID"]
    classes = df["Class_Label"]
    images_in_numpy = []
    for ID in images_IDs:
        image = Image.open(get_image_path(ID))
        images_in_numpy.append(np.array(image))

    classes_in_numpy = classes.to_numpy()
    len_train = int(len(classes_in_numpy) * 0.8)
    train_classes = classes_in_numpy[:len_train]
    train_images = images_in_numpy[:len_train]
    test_classes = classes_in_numpy[len_train:]
    test_images = images_in_numpy[len_train:]

    return (train_images, train_classes), (test_images, test_classes)

    
    # index = 1
    # np.set_printoptions(linewidth=320)
    # print(f'LABEL: {classes_in_numpy[index]}')
    # print(f'\nIMAGE PIXEL ARRAY:\n {images_in_numpy[index]}')

