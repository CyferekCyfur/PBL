import cv2 as cv
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

@tf.keras.utils.register_keras_serializable(package='Custom')
def orientation_loss(y_true, y_pred):
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

    loss = -(y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
    loss = tf.reduce_sum(loss, axis=1)
    epsilon = 1e-5
    anchors = anchors + epsilon
    loss = loss / anchors
    loss = tf.reduce_mean(loss)
    loss = 2 - 2 * loss

    return loss

# Function to calculate intersection of two boxes
def calculate_diff(box1, box2, original_size=(1224, 370), target_size=(224, 224)):

    # print("Box1:", box1)
    # print("Box2:", box2)
    x1, y1, _, w1, h1, _ = box1
    x2, y2, _, w2, h2, _ = box2

    # Get scaling factors
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]

    # Scale bounding box coordinates
    x1_scaled, y1_scaled, w1_scaled, h1_scaled = x1 * scale_x, y1 * scale_y, w1 * scale_x, h1 * scale_y
    x2_scaled, y2_scaled, w2_scaled, h2_scaled = x2 * scale_x, y2 * scale_y, w2 * scale_x, h2 * scale_y

    # Calculate intersection
    x_intersection = max(x1_scaled, x2_scaled)
    y_intersection = max(y1_scaled, y2_scaled)
    w_intersection = min(x1_scaled + w1_scaled, x2_scaled + w2_scaled) - x_intersection
    h_intersection = min(y1_scaled + h1_scaled, y2_scaled + h2_scaled) - y_intersection

    # Calculate areas of boxes
    area_box1 = w1_scaled * h1_scaled
    area_box2 = w2_scaled * h2_scaled
    area_intersection = max(0, w_intersection) * max(0, h_intersection)

    # Calculate difference
    diff = area_intersection / (area_box1 + area_box2 - area_intersection)
    return diff

# Function to read ground truth from a file
def read_ground_truth(image_number):
    image_number = image_number + '.txt'
    files_path = os.path.join(os.getcwd() + '/training/label_2/')
    
    if os.path.exists(files_path):
        with open(os.path.join(files_path, str(image_number)), 'r') as f:
            ground_truth_list = []
            for line in f:
                data = line.split()
                if len(data) > 1:
                    label = data[0]
                    # Parse data of object
                    truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, tx, ty, tz, ry = map(float, data[1:])
                    box = {
                        'label': label,
                        'truncated': truncated,
                        'occluded': occluded,
                        'alpha': alpha,
                        'bbox': (x1, y1, x2, y2),
                        'dimensions': (h, w, l),
                        'location': (tx, ty, tz),
                        'rotation_y': ry
                    }
                    ground_truth_list.append(box)
            return ground_truth_list
    else:
        return []

# Function to draw ground truth on an image
def draw_ground_truth(image, ground_truth_positions, original_size=(1224, 370), target_size=(224, 224)):
    # Get scaling factors
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]

    for position in ground_truth_positions:
        # Scale bounding box coordinates
        x1, y1, x2, y2 = position['bbox']
        x1_scaled, y1_scaled, x2_scaled, y2_scaled = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

        # Draw bounding box
        cv.rectangle(image, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)

        # Rescale coordinates of label
        label = f"{position['label']}: {position['alpha']:.2f}"
        cv.putText(image, label, (x1_scaled, y1_scaled - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main():
    # Read model
    model_path = os.path.join(os.getcwd() + '/YOLOV8-3D_models' + '/mobilenetv2/' + 'mobilenetv2_weights.h5')
    with tf.keras.utils.custom_object_scope({'orientation_loss': orientation_loss}):
        model = tf.keras.models.load_model(model_path)

    # Read images
    images_path = os.path.join(os.getcwd() + '/training/image_2/')
    images = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images.sort()
    # For each image in the folder make predictions, compare them with ground truth and show the image
    for image in images:
        img = cv.imread(os.path.join(images_path, image))
        img = cv.resize(img, (224, 224))
        predicted_positions = model.predict(tf.expand_dims(img, axis=0))
        decoded_predictions = decode_predictions(predicted_positions, top=3)[0]
        for i, prediction in enumerate(decoded_predictions):
            id, label, score = prediction
            print(f"{i + 1}: {label} ({100 * score:.2f})")
        ground_truth_positions = read_ground_truth(image[0:6])

        for gt_position in ground_truth_positions:
            draw_ground_truth(img, [gt_position])

        for pred_position in predicted_positions:
                for gt_position in ground_truth_positions:
                    diff = calculate_diff(pred_position, gt_position)
                print("Difference:", diff)
        
        # Show image with ground truth
        cv.imshow('Image with Ground Truth', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()
