import tensorflow as tf
import numpy as np



bbox_norm_value = 200


class IoUCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, model):
        super(IoUCallback, self).__init__()
        self.validation_data = validation_data
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        val_iou = calculate_iou(self.model.predict(self.validation_data[0])[0], self.validation_data[1])
        print(f'Epoch {epoch + 1} - Val IoU: {val_iou:.4f}\n')

def calculate_iou(y_pred, y_true):
    # Convert bounding box predictions to integer values
    y_pred = y_pred * bbox_norm_value
    y_true = y_true * bbox_norm_value
    y_pred = np.round(y_pred).astype(int)

    ious = [calculate_single_iou(pred_bbox, true_bbox)
            for pred_bbox, true_bbox in zip(y_pred, y_true)]

    average_iou = np.mean(ious)

    return average_iou

def calculate_single_iou(pred_bbox, true_bbox):
    
    x1 = max(pred_bbox[0], true_bbox[0])
    y1 = max(pred_bbox[1], true_bbox[1])
    x2 = min(pred_bbox[2], true_bbox[2])
    y2 = min(pred_bbox[3], true_bbox[3])

    width_intersection = max(0, x2 - x1)
    height_intersection = max(0, y2 - y1)
    area_intersection = width_intersection * height_intersection

    area_pred = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    area_true = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])

    area_union = area_pred + area_true - area_intersection


    iou = area_intersection / area_union if area_union > 0 else 0.0

    return iou

    

###############################################################################################################
###############################################################################################################
    

class CustomModel:
    def __init__(self):
        self.input_layer = tf.keras.Input(shape=(200, 200, 1))
        self.model = self.model_1()
        
    
    def backbone_1(self):
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(self.input_layer)
        maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
        conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(maxpool1)
        maxpool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
        flatten = tf.keras.layers.Flatten()(maxpool2)
        return flatten

    def model_1(self):
        
        input_from_backbone = self.backbone_1()
        
        bbox_fc1 = tf.keras.layers.Dense(64, activation='relu')(input_from_backbone)
        bbox_fc2 = tf.keras.layers.Dense(32, activation='relu')(bbox_fc1)
        bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox_output')(bbox_fc2)

        cls_fc1 = tf.keras.layers.Dense(64, activation='relu')(input_from_backbone)
        cls_fc2 = tf.keras.layers.Dense(32, activation='relu')(cls_fc1)
        cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(cls_fc2)

        return tf.keras.Model(inputs=self.input_layer, outputs=[bbox_output, cls_output])
    

    def model_2(self):
        input_from_backbone = self.backbone_1()
        
        bbox_fc1 = tf.keras.layers.Dense(128, activation='relu')(input_from_backbone)
        bbox_fc2 = tf.keras.layers.Dense(64, activation='relu')(bbox_fc1)
        bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox_output')(bbox_fc2)

        cls_fc1 = tf.keras.layers.Dense(128, activation='relu')(input_from_backbone)
        cls_fc2 = tf.keras.layers.Dense(64, activation='relu')(cls_fc1)
        cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(cls_fc2)

        return tf.keras.Model(inputs=self.input_layer, outputs=[bbox_output, cls_output])






    
