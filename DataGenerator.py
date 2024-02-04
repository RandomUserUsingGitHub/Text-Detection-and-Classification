import numpy as np
import keras
from PIL import Image
import pandas as pd


from DataLoader import load
from Generator.Utils.FileUtils import get_image_path, define_splitted_name

bbox_norm_value = 200
image_norm_value = 255


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, set_name, batch_size, output_size=(200,200), n_channels=1, shuffle=False):
        df = pd.read_csv(define_splitted_name(set_name))
        self.set_name = set_name
        self.images = df["Image_ID"]
        self.labels = df["Class_Label"]
        self.Bboxes = df["Bbox"]
        # print(self.Bboxes[0])
        self.n_channels = n_channels
        self.output_size = output_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.labels) / self.batch_size)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        # images = np.empty((self.batch_size, *self.output_size, self.n_channels))
        # bboxes = np.empty((self.batch_size, 4, 1))
        # labels = np.empty((self.batch_size), dtype=int)
        images = []
        bboxes = []
        labels = []

        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        # print("indices = ", indices)

        for i, data_index in enumerate(indices):
            image = Image.open(get_image_path(self.images[data_index]))
            image = image.resize(self.output_size)
            
            # images[data_index] = (np.array(image))
            # bboxes[data_index] = (self.Bboxes[data_index])
            # labels[data_index] = (self.labels[data_index])
            
            images.append(np.array(image))
            bboxes.append(self.Bboxes[data_index])
            labels.append(self.labels[data_index])

        Bboxes_in_integer = [[int(integer) for integer in bbox.split(',')] for bbox in bboxes]

        images_in_numpy = np.array(images, dtype=float)
        images_norm = images_in_numpy / image_norm_value
        labels_in_numpy = np.array(labels)
        bboxes_in_numpy = np.array(Bboxes_in_integer) / bbox_norm_value


        # print("name = ", self.set_name)
        # print("index = ", idx)
        # print("-> labels: ", labels_in_numpy)
        # print("-> type labels: ", type(labels_in_numpy))
        # print("-> bbox: ", bboxes_in_numpy)
        # print("-> type bbox: ", type(bboxes_in_numpy))
        # print("-----------------------------------")

        # return images_norm, bboxes_in_numpy, labels_in_numpy
        return (images_norm, {'bbox_output': bboxes_in_numpy, 'cls_output': labels_in_numpy})