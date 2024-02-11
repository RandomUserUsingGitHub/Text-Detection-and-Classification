import tensorflow as tf
from models import calculate_iou, calculate_single_iou, giou_loss
from Generator.Utils.FileUtils import get_model_path, list_models
from DataLoader import load
import cv2
import numpy as np
import time
from DataGenerator import DataGenerator


bbox_norm_value = 200
batch_size = 20



def visualize_bboxes(image, true_bbox, pred_bbox, prediction_label):
    image = (image * 255).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    true_bbox = np.array(true_bbox, dtype=int)
    pred_bbox = np.array(pred_bbox, dtype=int)

    print("true: ", true_bbox)
    print("pred: ", pred_bbox)

    x1, y1, x2, y2 = true_bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    x1, y1, x2, y2 = pred_bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 150), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if prediction_label == 0:
        cv2.putText(image, 'English', (x1, y1 - 10), font, 0.3, (0, 0, 0), 1,  cv2.LINE_AA)
    elif prediction_label == 1:
        cv2.putText(image, 'Persian', (x1, y1 - 10), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('Image with Bounding Boxes', image)
    
            


def accuracy(model, images, true_bboxes, true_labels):
    for i in range(len(images)):
        reshaped_image = np.expand_dims(images[i], axis=0)
        start = time.time()
        bbox, prediction = model.predict(reshaped_image)
        end = time.time()


        binary_prediction = 1 if prediction[0] >= 0.5 else 0

        print(f"True Label: {true_labels[i]}, Predicted Label: {binary_prediction}")
        # print(f"True Bbox: {true_bboxes[i] * bbox_norm_value}, Predicted Bbox: {bbox[0] * bbox_norm_value}")
        # print("-> Prediction time = ", (end - start)*1000, " ms\n")
        visualize_bboxes(images[i], true_bboxes[i] * bbox_norm_value, bbox[0] * bbox_norm_value, binary_prediction)
        print(f"IoU = {calculate_single_iou((true_bboxes[i] * bbox_norm_value), np.array(bbox[0]) * bbox_norm_value)}\n")
        key = cv2.waitKey(0)
        if key == ord('q') or key == ord('Q'):
            return False
        return True
    

def evaluate_model():
    # (test_images, test_labels, test_Bboxes) = load("test")
    test_gen = DataGenerator("test", batch_size=batch_size)



    files = []
    print("List of models: ")
    for i, file in enumerate(list_models()):
        print(f'{i+1}. {file}')
        files.append(file)
    inp = int(input("-> "))
    name = files[inp-1]
    path = get_model_path(name)
    model = tf.keras.models.load_model(path, custom_objects={'giou_loss': giou_loss})

    model.evaluate(test_gen)

    ious = []
    for i in range(len(test_gen)):
        batch_data = test_gen[i]
        images = batch_data[0]
        outputs = batch_data[1]

        true_bboxes = outputs['bbox_output']
        true_labels = outputs['cls_output']
        y_pred = model.predict(images, verbose=0)
        print(f"{i+1}/{len(test_gen)}")
        ious.append(calculate_iou(true_bboxes * bbox_norm_value, y_pred[0] * bbox_norm_value))

    print("Bbox accuracy = ", np.mean(ious))

 
    for i in range(len(test_gen)):
        batch_data = test_gen[i]
        images = batch_data[0]
        outputs = batch_data[1]

        true_bboxes = outputs['bbox_output']
        true_labels = outputs['cls_output']
        if accuracy(model, images, true_bboxes, true_labels) == False:
            break
    cv2.destroyAllWindows()


