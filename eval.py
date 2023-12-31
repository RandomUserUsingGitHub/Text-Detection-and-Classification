import tensorflow as tf
from models import calculate_iou
from Generator.Utils.FileUtils import get_model_path, list_models
from DataLoader import get_test_data
import cv2
import numpy as np


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
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def evaluate_model():
    (test_images, test_labels, test_Bboxes) = get_test_data()

    files = []
    print("List of models: ")
    for i, file in enumerate(list_models()):
        print(f'{i+1}. {file}')
        files.append(file)
    inp = int(input("-> "))
    name = files[inp-1]
    path = get_model_path(name)
    model = tf.keras.models.load_model(path)

    eval_results = model.evaluate(np.array(test_images), {'bbox_output': np.array(test_Bboxes), 'cls_output': np.array(test_labels)})

    y_pred = model.predict(test_images)

    iou = calculate_iou(test_Bboxes, y_pred[0])

    print("Bbox accuracy = ", iou)

    for i in range(len(test_images)):
        reshaped_image = np.expand_dims(test_images[i], axis=0)
        bbox, prediction = model.predict(reshaped_image)

        binary_prediction = 1 if prediction[0] >= 0.5 else 0

        print(f"\nSample {i + 1} - True Label: {test_labels[i]}, Predicted Label: {binary_prediction}")
        print(f"True Bbox: {test_Bboxes[i]}, Predicted Bbox: {bbox[0]}")

        visualize_bboxes(test_images[i], test_Bboxes[i], bbox[0], binary_prediction)

