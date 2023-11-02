import numpy as np
import pandas as pd

from PIL import Image
from Generator.Utils.FileUtils import define_csv_name, get_image_path

def read_data():
    df = pd.read_csv(define_csv_name())
    images_IDs = df["Image_ID"]
    classes = df["Class_Label"]
    images = []
    for ID in images_IDs:
        image = Image.open(get_image_path(ID))
        images.append(image)

    
    images_in_numpy = np.array(images)
    classes_in_numpy = classes.to_numpy()

    index = 0

    # Set number of characters per row when printing
    np.set_printoptions(linewidth=320)

    # Print the label and image
    print(f'LABEL: {classes_in_numpy[index]}')
    print(f'\nIMAGE PIXEL ARRAY:\n {images_in_numpy[index]}')

