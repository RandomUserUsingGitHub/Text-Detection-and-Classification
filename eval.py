import tensorflow as tf
from models import calculate_iou, calculate_single_iou
from Generator.Utils.FileUtils import get_model_path, list_models
from DataLoader import load
import cv2
import numpy as np
import time

bbox_norm_value = 200


def visualize_bboxes(image, true_bbox, pred_bbox, prediction_label):
    image = (image * 255).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    true_bbox = np.array(true_bbox, dtype=int)
    pred_bbox = np.array(pred_bbox, dtype=int)

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
    
            
        
    

def evaluate_model():
    (test_images, test_labels, test_Bboxes) = load("test")

    files = []
    print("List of models: ")
    for i, file in enumerate(list_models()):
        print(f'{i+1}. {file}')
        files.append(file)
    inp = int(input("-> "))
    name = files[inp-1]
    path = get_model_path(name)
    model = tf.keras.models.load_model(path)

    model.evaluate(np.array(test_images), {'bbox_output': np.array(test_Bboxes), 'cls_output': np.array(test_labels)})

    y_pred = model.predict(test_images)

    iou = calculate_iou(test_Bboxes * bbox_norm_value, y_pred[0] * bbox_norm_value)

    print("Bbox accuracy = ", iou)

    # total_time = []
    # for i in range(len(test_images)):
    #     reshaped_image = np.expand_dims(test_images[i], axis=0)
    #     start = time.time()
    #     bbox, prediction = model.predict(reshaped_image)
    #     end = time.time()

    #     if(i > 0):
    #         total_time.append(end - start)
    #     print(f"Sample {i+1}/{len(test_images)}")

    # average_time = np.mean(total_time)
    # print("-> Average time = ", (average_time)*1000, " ms")
    
    
    for i in range(len(test_images)):
        reshaped_image = np.expand_dims(test_images[i], axis=0)
        start = time.time()
        bbox, prediction = model.predict(reshaped_image)
        print(bbox[0])
        print(test_Bboxes[i])
        end = time.time()


        binary_prediction = 1 if prediction[0] >= 0.5 else 0

        print(f"Sample {i + 1}:\nTrue Label: {test_labels[i]}, Predicted Label: {binary_prediction}")
        print(f"True Bbox: {test_Bboxes[i] * bbox_norm_value}, Predicted Bbox: {bbox[0] * bbox_norm_value}")
        print(f"IoU = {calculate_single_iou((test_Bboxes[i] * bbox_norm_value), np.array(bbox[0]) * bbox_norm_value)}")
        print("-> Prediction time = ", (end - start)*1000, " ms\n")

        visualize_bboxes(test_images[i], test_Bboxes[i] * bbox_norm_value, bbox[0] * bbox_norm_value, binary_prediction)
        key = cv2.waitKey(0)
        if key == ord('q') or key == ord('Q'):
            return
    cv2.destroyAllWindows()


