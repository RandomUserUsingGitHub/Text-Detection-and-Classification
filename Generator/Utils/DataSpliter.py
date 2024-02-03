import numpy as np
import pandas as pd

from Generator.Utils.FileUtils import define_csv_name, get_image_path, define_splitted_name


train_data_percentage = 0.8
validation_data_percentage = 0.1
test_data_percentage = 0.1

def split_data():
    df = pd.read_csv(define_csv_name())
    images_IDs = df["Image_ID"]
    labels = df["Class_Label"]
    Bboxes = df["Bbox"]

    len_dataset = int(len(labels))
    len_train = int(len_dataset * train_data_percentage)
    len_validation = int(len_dataset * validation_data_percentage)

    train_images_IDs = images_IDs[:len_train]
    train_labels = labels[:len_train]
    train_Bboxes = Bboxes[:len_train]
    data_dict = {
        'Image_ID': train_images_IDs,
        'Bbox': train_Bboxes,
        'Class_Label': train_labels
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(define_splitted_name("train"), index=False, sep=",")

    validation_images_IDs = images_IDs[len_train:len_train+len_validation]
    validation_labels = labels[len_train:len_train+len_validation]
    validation_Bboxes = Bboxes[len_train:len_train+len_validation]
    data_dict = {
        'Image_ID': validation_images_IDs,
        'Bbox': validation_Bboxes,
        'Class_Label': validation_labels}
    df = pd.DataFrame(data_dict)
    df.to_csv(define_splitted_name("validation"), index=False, sep=",")

    test_images_IDs = images_IDs[len_train+len_validation:len_dataset]
    test_labels = labels[len_train+len_validation:len_dataset]
    test_Bboxes = Bboxes[len_train+len_validation:len_dataset]
    data_dict = {
        'Image_ID': test_images_IDs,
        'Bbox': test_Bboxes,
        'Class_Label': test_labels
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(define_splitted_name("test"), index=False, sep=",")

