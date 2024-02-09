import numpy as np
import pandas as pd

from PIL import Image
from Generator.Utils.FileUtils import get_image_path, define_splitted_name


bbox_norm_value = 200
image_norm_value = 255


test_images = []
test_labels = []
test_Bboxes = []


def load(set_name):
    df = pd.read_csv(define_splitted_name(set_name))
    images_IDs = df["Image_ID"]
    labels = df["Class_Label"]
    Bboxes = df["Bbox"]
    images_in_numpy = []
    for ID in images_IDs:
        image = Image.open(get_image_path(ID))
        image = image.resize((200, 200))
        images_in_numpy.append(np.array(image))

    print(Bboxes, type(Bboxes))

    Bboxes_in_integer = [[int(integer) for integer in bbox.split(',')] for bbox in Bboxes]
    Bboxes_in_numpy = np.array(Bboxes_in_integer)
    labels_in_numpy = labels.to_numpy()

    images_in_numpy = np.array(images_in_numpy, dtype=float)    
    images_norm = images_in_numpy / image_norm_value
    bboxes_norm = Bboxes_in_numpy / bbox_norm_value


    return images_norm, labels_in_numpy, bboxes_norm

