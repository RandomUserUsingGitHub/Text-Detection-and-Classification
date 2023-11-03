import numpy as np
import pandas as pd

from PIL import Image
from Generator.Utils.FileUtils import define_csv_name, get_image_path

train_data_percentage = 0.7

def read_data():
    df = pd.read_csv(define_csv_name())
    images_IDs = df["Image_ID"]
    labels = df["Class_Label"]
    images_in_numpy = []
    for ID in images_IDs:
        image = Image.open(get_image_path(ID))
        image = image.resize((200, 200))
        images_in_numpy.append(np.array(image))

    labels_in_numpy = labels.to_numpy()


    len_dataset = int(len(labels_in_numpy) / 2)
    len_train = int(len_dataset * train_data_percentage)
    print(len_dataset)
    print(len_train)

    

#   eng
    train_images_eng = images_in_numpy[:len_train]
    test_images_eng = images_in_numpy[len_train:len_dataset]
    train_labels_eng = labels_in_numpy[:len_train]
    test_labels_eng = labels_in_numpy[len_train:len_dataset]


#   pes
    train_images_pes = images_in_numpy[len_dataset:len_train + len_dataset]
    test_images_pes = images_in_numpy[len_train + len_dataset:]
    train_labels_pes = labels_in_numpy[len_dataset:len_train + len_dataset]
    test_labels_pes = labels_in_numpy[len_train + len_dataset:]

#   merge
    #   eng
    train_images = train_images_eng + train_images_pes
    test_images = test_images_eng + test_images_pes
    train_labels = np.concatenate((train_labels_eng, train_labels_pes))
    test_labels = np.concatenate((test_labels_eng, test_labels_pes))

    return (train_images, train_labels), (test_images, test_labels)

    
    # index = 1
    # np.set_printoptions(linewidth=320)
    # print(f'LABEL: {labels_in_numpy[index]}')
    # print(f'\nIMAGE PIXEL ARRAY:\n {images_in_numpy[index]}')

