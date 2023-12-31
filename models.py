import tensorflow as tf
import numpy as np


class IoUCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=None):
        super(IoUCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_iou = calculate_iou(self.model.predict(self.validation_data[0])[0], self.validation_data[1])
        print(f'Epoch {epoch + 1} - Val IoU: {val_iou:.4f}')

def calculate_iou(y_pred, y_true):
    # Convert bounding box predictions to integer values
    y_pred = np.round(y_pred).astype(int)

    # Calculate IoU for each sample
    ious = [calculate_single_iou(pred_bbox, true_bbox)
            for pred_bbox, true_bbox in zip(y_pred, y_true)]

    # Average IoU across all samples
    average_iou = np.mean(ious)

    return average_iou

def calculate_single_iou(pred_bbox, true_bbox):
    # Extract coordinates
    pred_x, pred_y, pred_w, pred_h = pred_bbox
    true_x, true_y, true_w, true_h = true_bbox

    # Calculate intersection coordinates
    x1 = max(pred_x, true_x)
    y1 = max(pred_y, true_y)
    x2 = min(pred_x + pred_w, true_x + true_w)
    y2 = min(pred_y + pred_h, true_y + true_h)

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    pred_area = pred_w * pred_h
    true_area = true_w * true_h
    union_area = pred_area + true_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

    

###############################################################################################################
###############################################################################################################
    

class CustomModel:
    def __init__(self, model_type):
        self.input_layer = tf.keras.Input(shape=(200, 200, 1))
        self.model_type = model_type
        self.model = self.build_model()

    def build_model(self):
        if self.model_type == 'model1':
            return self.model_1()
        elif self.model_type == 'model2':
            return self.model_2()
        elif self.model_type == 'custom2':
            return self.build_custom_model2()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
    
    def backbone_1(self):
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(self.input_layer)
        maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool1)
        maxpool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
        flatten = tf.keras.layers.Flatten()(maxpool2)
        return flatten

    def model_1(self):
        
        input_from_backbone = self.backbone_1()
        
        bbox_fc1 = tf.keras.layers.Dense(64, activation='relu')(input_from_backbone)
        bbox_fc2 = tf.keras.layers.Dense(32, activation='relu')(bbox_fc1)
        bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox_output')(bbox_fc2)  # 4 for (x, y, w, h)

        cls_fc1 = tf.keras.layers.Dense(64, activation='relu')(input_from_backbone)
        cls_fc2 = tf.keras.layers.Dense(32, activation='relu')(cls_fc1)
        cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(cls_fc2)

        return tf.keras.Model(inputs=self.input_layer, outputs=[bbox_output, cls_output])
    

    def model_2(self):
        input_from_backbone = self.backbone_1()
        
        bbox_fc1 = tf.keras.layers.Dense(64, activation='relu')(input_from_backbone)
        bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox_output')(bbox_fc1)  # 4 for (x, y, w, h)

        cls_fc1 = tf.keras.layers.Dense(32, activation='relu')(input_from_backbone)
        cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(cls_fc1)

        return tf.keras.Model(inputs=self.input_layer, outputs=[bbox_output, cls_output])


    import numpy as np




    
