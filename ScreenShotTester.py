import tensorflow as tf
from models import calculate_iou, calculate_single_iou, giou_loss
from Generator.Utils.FileUtils import get_model_path, list_models
from DataLoader import load
import cv2
import numpy as np
import time
import pyperclip
from PIL import ImageGrab
from io import BytesIO
from PIL import Image
import base64



def visualize_bboxes(image, pred_bbox, prediction_label):
    print("vis")
    image = np.squeeze(image, axis=0)
    image = (image * 255).astype(np.uint8)

    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    pred_bbox = np.array(pred_bbox, dtype=int)
    print(pred_bbox)
    print(image.shape)

    x1, y1, x2, y2 = pred_bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 150), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if prediction_label == 0:
        cv2.putText(image, 'English', (x1, y1 - 10), font, 0.3, (0, 0, 0), 1,  cv2.LINE_AA)
    elif prediction_label == 1:
        cv2.putText(image, 'Persian', (x1, y1 - 10), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('Image with Bounding Boxes', image)
    


def read_clipboard(model):
    Flag = True
    while Flag:
        try:
            screenshot = ImageGrab.grabclipboard()
            print("Original image shape:", screenshot.size)
            image = screenshot.resize((200,200))
            print("Resized image shape:", image.size)
            image = image.convert("L")
            image = np.array(image, dtype=float)
            image = image / 255.0
            image = np.expand_dims(image, axis=-1)
            print("after first: ", image.shape)
            image = np.expand_dims(image, axis=0)
            print("Processed image shape:", image.shape)
            

            
            # reshaped_image = np.expand_dims(image, axis=0)
            bbox, prediction = model.predict(image)

            binary_prediction = 1 if prediction[0] >= 0.5 else 0

            visualize_bboxes(image, bbox[0] * 200, binary_prediction)
            key = cv2.waitKey(0) % 256
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                Flag = False 

        except Exception as e:
            print("Error capturing screen:", e)




def test():
    files = []
    print("List of models: ")
    for i, file in enumerate(list_models()):
        print(f'{i+1}. {file}')
        files.append(file)
    inp = int(input("-> "))
    name = files[inp-1]
    path = get_model_path(name)
    model = tf.keras.models.load_model(path, custom_objects = {'giou_loss': giou_loss})

    read_clipboard(model)
    cv2.destroyAllWindows()





