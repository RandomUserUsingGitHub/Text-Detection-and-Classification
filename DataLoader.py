import numpy as np
import pandas as pd

from PIL import Image
from Generator.Utils.FileUtils import define_csv_name, get_image_path

train_data_percentage = 0.7

def read_data():
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

    labels_in_numpy = labels.to_numpy()
    # Bboxes_in_numpy = Bboxes.to_numpy()
    Bboxes_in_numpy = Bboxes_in_integer
    print(type(Bboxes_in_integer[0]))

    len_dataset = int(len(labels_in_numpy) / 2)
    len_train = int(len_dataset * train_data_percentage)
    print(len_dataset)
    print(len_train)

    

#   eng
    train_images_eng = images_in_numpy[:len_train]
    test_images_eng = images_in_numpy[len_train:len_dataset]
    train_labels_eng = labels_in_numpy[:len_train]
    test_labels_eng = labels_in_numpy[len_train:len_dataset]
    train_Bboxes_eng = Bboxes_in_numpy[:len_train]
    test_Bboxes_eng = Bboxes_in_numpy[len_train:len_dataset]


#   pes
    train_images_pes = images_in_numpy[len_dataset:len_train + len_dataset]
    test_images_pes = images_in_numpy[len_train + len_dataset:]
    train_labels_pes = labels_in_numpy[len_dataset:len_train + len_dataset]
    test_labels_pes = labels_in_numpy[len_train + len_dataset:]
    train_Bboxes_pes = Bboxes_in_numpy[len_dataset:len_train + len_dataset]
    test_Bboxes_pes = Bboxes_in_numpy[len_train + len_dataset:]

#   merge
    #   eng
    train_images = train_images_eng + train_images_pes
    test_images = test_images_eng + test_images_pes
    train_labels = np.concatenate((train_labels_eng, train_labels_pes))
    test_labels = np.concatenate((test_labels_eng, test_labels_pes))
    train_Bboxes = np.concatenate((train_Bboxes_eng, train_Bboxes_pes))
    test_Bboxes = np.concatenate((test_Bboxes_eng, test_Bboxes_pes))

    return (train_images, train_labels, train_Bboxes), (test_images, test_labels, test_Bboxes)

    
    # index = 1
    # np.set_printoptions(linewidth=320)
    # print(f'LABEL: {labels_in_numpy[index]}')
    # print(f'\nIMAGE PIXEL ARRAY:\n {images_in_numpy[index]}')

