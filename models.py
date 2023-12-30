import tensorflow as tf
import numpy as np

class IoUCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data=None):
        super(IoUCallback, self).__init__()
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        # Calculate IoU for training set
        train_iou = self.calculate_iou(self.validation_data[0], self.validation_data[1])
        print(f'\nEpoch {epoch + 1} - Train IoU: {train_iou:.4f}')

        # Calculate IoU for validation set
        val_iou = self.calculate_iou(self.validation_data[0], self.validation_data[1])
        print(f'Epoch {epoch + 1} - Val IoU: {val_iou:.4f}')

    def on_test_end(self, logs=None):
        # Set the test_data attribute during the evaluation process
        self.test_data = (self.validation_data[0], self.validation_data[1])

        # Calculate IoU for the test set
        test_iou = self.calculate_iou(self.test_data[0], self.test_data[1])
        print(f'\nTest IoU: {test_iou:.4f}')

    def calculate_iou(self, images, true_bboxes):
        # Predict bounding boxes using the model
        predicted_bboxes = self.model.predict(images)

        # Calculate IoU for each sample
        ious = [self.calculate_single_iou(true_bbox, pred_bbox)
                for true_bbox, pred_bbox in zip(true_bboxes, predicted_bboxes[0])]

        # Average IoU across all samples
        average_iou = np.mean(ious)

        return average_iou

    @staticmethod
    def calculate_single_iou(true_bbox, pred_bbox):
        # Extract coordinates
        true_x, true_y, true_w, true_h = true_bbox
        pred_x, pred_y, pred_w, pred_h = pred_bbox

        # Calculate intersection coordinates
        x1 = max(true_x, pred_x)
        y1 = max(true_y, pred_y)
        x2 = min(true_x + true_w, pred_x + pred_w)
        y2 = min(true_y + true_h, pred_y + pred_h)

        # Calculate intersection area
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        true_area = true_w * true_h
        pred_area = pred_w * pred_h
        union_area = true_area + pred_area - intersection_area

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
        
        # Bounding Box Regression Branch
        bbox_fc1 = tf.keras.layers.Dense(64, activation='relu')(input_from_backbone)
        bbox_fc2 = tf.keras.layers.Dense(32, activation='relu')(bbox_fc1)
        bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox_output')(bbox_fc2)  # 4 for (x, y, w, h)

        # Binary Classification Branch
        cls_fc1 = tf.keras.layers.Dense(64, activation='relu')(input_from_backbone)
        cls_fc2 = tf.keras.layers.Dense(32, activation='relu')(cls_fc1)
        cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(cls_fc2)

        return tf.keras.Model(inputs=self.input_layer, outputs=[bbox_output, cls_output])
    

    def model_2(self):
        input_from_backbone = self.backbone_1()
        
        # Bounding Box Regression Branch
        bbox_fc1 = tf.keras.layers.Dense(128, activation='relu')(input_from_backbone)
        bbox_fc2 = tf.keras.layers.Dense(64, activation='relu')(bbox_fc1)
        bbox_fc3 = tf.keras.layers.Dense(32, activation='relu')(bbox_fc2)
        bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox_output')(bbox_fc3)  # 4 for (x, y, w, h)

        # Binary Classification Branch
        cls_fc1 = tf.keras.layers.Dense(128, activation='relu')(input_from_backbone)
        cls_fc2 = tf.keras.layers.Dense(64, activation='relu')(cls_fc1)
        cls_fc3 = tf.keras.layers.Dense(32, activation='relu')(cls_fc2)
        cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(cls_fc2)

        return tf.keras.Model(inputs=self.input_layer, outputs=[bbox_output, cls_output])


    import numpy as np

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    area_intersection = w_intersection * h_intersection

    area_union = (w1 * h1) + (w2 * h2) - area_intersection

    iou = area_intersection / max(area_union, 1e-10)  # Avoid division by zero

    return iou


    
