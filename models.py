import tensorflow as tf
import numpy as np



bbox_norm_value = 200




def giou_loss(y_true, y_pred):
    """
    Calculate Generalized Intersection over Union (GIoU) loss between predicted and ground truth bounding boxes.
    
    Arguments:
    y_true -- true bounding boxes, tensor of shape (batch_size, 4)
    y_pred -- predicted bounding boxes, tensor of shape (batch_size, 4)
    
    Returns:
    GIoU loss
    """
    # Extract coordinates of true and predicted boxes
    true_x1, true_y1, true_x2, true_y2 = tf.split(y_true, 4, axis=-1)
    pred_x1, pred_y1, pred_x2, pred_y2 = tf.split(y_pred, 4, axis=-1)
    
    # Calculate intersection coordinates
    inter_x1 = tf.maximum(true_x1, pred_x1)
    inter_y1 = tf.maximum(true_y1, pred_y1)
    inter_x2 = tf.minimum(true_x2, pred_x2)
    inter_y2 = tf.minimum(true_y2, pred_y2)
    
    # Calculate intersection area
    inter_area = tf.maximum(inter_x2 - inter_x1, 0) * tf.maximum(inter_y2 - inter_y1, 0)
    
    # Calculate union area
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    union_area = true_area + pred_area - inter_area
    
    # Calculate GIoU
    iou = inter_area / tf.maximum(union_area, 1e-6)
    enclose_x1 = tf.minimum(true_x1, pred_x1)
    enclose_y1 = tf.minimum(true_y1, pred_y1)
    enclose_x2 = tf.maximum(true_x2, pred_x2)
    enclose_y2 = tf.maximum(true_y2, pred_y2)
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    giou = iou - (enclose_area - union_area) / tf.maximum(enclose_area, 1e-6)
    
    # Return GIoU loss
    return 1 - giou


###############################################################################################################
###############################################################################################################


class CustomWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, bbox_weight, cls_weight, **kwargs):
        super().__init__(**kwargs)
        self.bbox_weight = bbox_weight
        self.cls_weight = cls_weight

    def call(self, y_true, y_pred):
        # Define your binary cross-entropy loss
        cls_loss = tf.keras.losses.binary_crossentropy(y_true[1], y_pred[1])

        # Define your mean squared error loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true[0], y_pred[0])

        # Apply weights
        weighted_loss = (cls_loss * self.cls_weight) + (mse_loss * self.bbox_weight)

        # Return the combined loss
        return weighted_loss


###############################################################################################################
###############################################################################################################


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


class IoUCallback2(tf.keras.callbacks.Callback):
    def __init__(self, generator):
        super(IoUCallback2, self).__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        # Compute IoU on the validation set
        iou = self.compute_iou()
        
        # Log the IoU value
        logs["iou"] = iou
        print(f"IoU: {iou}")

    def compute_iou(self):
        iou_values = []
        for batch in range(len(self.generator)):
            batch_data = self.generator[batch]

            
            images = batch_data[0]
            outputs = batch_data[1]

            true_bboxes = outputs['bbox_output']
            true_labels = outputs['cls_output']
            
            predicted_results = self.model.predict(images, verbose=0)

            # If you have separate arrays for bounding boxes and labels, you can unpack them
            predicted_bboxes, predicted_labels = predicted_results
            
            
            # Compute IoU for each example in the batch
            batch_iou = self.compute_batch_iou(true_bboxes, predicted_bboxes)
            iou_values.append(batch_iou)
        
        # Average IoU over all batches
        mean_iou = np.mean(iou_values)
        return mean_iou

    def compute_batch_iou(self, true_bboxes, predicted_bboxes):
        # Extract coordinates from bounding boxes
        x1_true, y1_true, x2_true, y2_true = true_bboxes[:, 0], true_bboxes[:, 1], true_bboxes[:, 2], true_bboxes[:, 3]
        x1_pred, y1_pred, x2_pred, y2_pred = predicted_bboxes[:, 0], predicted_bboxes[:, 1], predicted_bboxes[:, 2], predicted_bboxes[:, 3]

        # Calculate intersection area
        intersection_area = np.maximum(0.0, np.minimum(x2_true, x2_pred) - np.maximum(x1_true, x1_pred)) * np.maximum(0.0, np.minimum(y2_true, y2_pred) - np.maximum(y1_true, y1_pred))

        # Calculate union area
        area_true = (x2_true - x1_true) * (y2_true - y1_true)
        area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        union_area = area_true + area_pred - intersection_area

        # Calculate IoU
        iou = intersection_area / np.maximum(union_area, 1e-10)  # Avoid division by zero
        mean_batch_iou = np.mean(iou)

        return mean_batch_iou


    

###############################################################################################################
###############################################################################################################
    

class CustomModel:
    def __init__(self):
        self.input_layer = tf.keras.Input(shape=(200, 200, 1))
        self.model = self.model_5()
        
    
    def backbone_1(self):
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(self.input_layer)
        maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
        conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(maxpool1)
        maxpool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
        flatten = tf.keras.layers.Flatten()(maxpool2)
        return flatten


    def backbone_2(self):
        x = self.input_layer

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Flatten()(x)
        return x
    
    
    def backbone_3(self):
        x = self.input_layer

        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)

        x = tf.keras.layers.Flatten()(x)
        return x
    

    def backbone_4(self):
        x = self.input_layer

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)


        x = tf.keras.layers.Flatten()(x)
        return x
    

    def backbone_5(self):
        x = self.input_layer

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3))(x)


        x = tf.keras.layers.Flatten()(x)
        return x
    
    

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
    

    def model_3(self):
        input_from_backbone = self.backbone_3()
        
        bbox_fc1 = tf.keras.layers.Dense(256, activation='relu')(input_from_backbone)
        bbox_fc2 = tf.keras.layers.Dense(128, activation='relu')(bbox_fc1)
        bbox_fc3 = tf.keras.layers.Dense(64, activation='relu')(bbox_fc2)
        bbox_fc4 = tf.keras.layers.Dense(32, activation='relu')(bbox_fc3)   
        bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox_output')(bbox_fc4)

        cls_fc1 = tf.keras.layers.Dense(64, activation='relu')(input_from_backbone)
        cls_fc2 = tf.keras.layers.Dense(32, activation='relu')(cls_fc1)
        cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(cls_fc2)

        return tf.keras.Model(inputs=self.input_layer, outputs=[bbox_output, cls_output])
    

    def model_4(self):
        input_from_backbone = self.backbone_4()
        
        bbox_fc1 = tf.keras.layers.Dense(256, activation='relu')(input_from_backbone)
        bbox_fc2 = tf.keras.layers.Dense(256, activation='relu')(bbox_fc1)
        bbox_fc3 = tf.keras.layers.Dense(128, activation='relu')(bbox_fc2)
        bbox_fc4 = tf.keras.layers.Dense(64, activation='relu')(bbox_fc3)   
        bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox_output')(bbox_fc4)

        cls_fc1 = tf.keras.layers.Dense(64, activation='relu')(input_from_backbone)
        cls_fc2 = tf.keras.layers.Dense(32, activation='relu')(cls_fc1)
        cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(cls_fc2)

        return tf.keras.Model(inputs=self.input_layer, outputs=[bbox_output, cls_output])
    

    def model_5(self):
        input_from_backbone = self.backbone_3()
        
        bbox_fc1 = tf.keras.layers.Dense(256, activation='relu')(input_from_backbone)
        bbox_fc2 = tf.keras.layers.Dense(128, activation='relu')(bbox_fc1)
        bbox_fc3 = tf.keras.layers.Dense(256, activation='relu')(bbox_fc2)
        bbox_fc4 = tf.keras.layers.Dense(128, activation='relu')(bbox_fc3)
        bbox_fc5 = tf.keras.layers.Dense(16, activation='relu')(bbox_fc4)   
        bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox_output')(bbox_fc5)

        cls_fc1 = tf.keras.layers.Dense(64, activation='relu')(input_from_backbone)
        cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(cls_fc1)

        return tf.keras.Model(inputs=self.input_layer, outputs=[bbox_output, cls_output])


    def model_6(self):
        input_from_backbone = self.backbone_5()
        
        x = tf.keras.layers.Dense(128, activation='relu')(input_from_backbone)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(16, activation='relu')(x)   
        bbox_output = tf.keras.layers.Dense(4, activation='linear', name='bbox_output')(x)

        x = tf.keras.layers.Dense(32, activation='relu')(input_from_backbone)
        cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(x)

        return tf.keras.Model(inputs=self.input_layer, outputs=[bbox_output, cls_output])



    
