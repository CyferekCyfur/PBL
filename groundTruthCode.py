import cv2 as cv
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join

# Function to calculate intersection of two boxes
def calculate_diff(box1, box2):
    if len(box1) == 6 and len(box2) == 6:
        x1, y1, _, w1, h1, _ = box1
        x2, y2, _, w2, h2, _ = box2
    else:
        return 0

    # Calculate intersection coordinates
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection

    # Calculate areas of boxes and intersection
    area_box1 = w1 * h1
    area_box2 = w2 * h2
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
def draw_ground_truth(image, ground_truth_positions):
    for position in ground_truth_positions:
        x1, y1, x2, y2 = map(int, position['bbox'])
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{position['label']}: {position['alpha']:.2f}"
        cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    # Read model
    model_path = os.path.join(os.getcwd() + '/YOLOV8-3D_models' + '/mobilenetv2/' + 'mobilenetv2_weights.h5')
    model_path = '/home/jakub/Documents/PBL/mobilenetv2_weights.h5'
    model = tf.keras.applications.MobileNetV2(weights=None)
    model.load_weights(model_path, by_name=True)

    images_path = os.path.join(os.getcwd() + '/training/image_2/')
    images = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images.sort()
    i = 0
    for image in images:
        i += 1
        img = cv.imread(os.path.join(images_path, image))
        img = cv.resize(img, (224, 224))
        predicted_positions = model.predict(tf.expand_dims(img, axis=0))
        # print(f'photo: {image}, positions: {predicted_positions}')
        ground_truth_positions = read_ground_truth(image[0:6])

        for gt_position in ground_truth_positions:
            draw_ground_truth(img, [gt_position])

        for pred_position in predicted_positions:
                for gt_position in ground_truth_positions:
                    diff = calculate_diff(pred_position, gt_position)
                print("Difference:", diff)
        
        cv.imshow('Image with Ground Truth', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        if i == 10:
            break


if __name__ == '__main__':
    main()
