import cv2 as cv
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Function to calculate intersection of two boxes
def calculate_diff(box1, box2, original_size=(1224, 370), target_size=(224, 224)):

    # print("Box1:", box1)
    # print("Box2:", box2)
    if len(box1) == 6 and len(box2) == 6:
        x1, y1, _, w1, h1, _ = box1
        x2, y2, _, w2, h2, _ = box2
    else:
        return 0

    # Pobierz współczynniki skalowania
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]

    # Przeskaluj współrzędne bounding boxów
    x1_scaled, y1_scaled, w1_scaled, h1_scaled = x1 * scale_x, y1 * scale_y, w1 * scale_x, h1 * scale_y
    x2_scaled, y2_scaled, w2_scaled, h2_scaled = x2 * scale_x, y2 * scale_y, w2 * scale_x, h2 * scale_y

    # Oblicz współrzędne intersection
    x_intersection = max(x1_scaled, x2_scaled)
    y_intersection = max(y1_scaled, y2_scaled)
    w_intersection = min(x1_scaled + w1_scaled, x2_scaled + w2_scaled) - x_intersection
    h_intersection = min(y1_scaled + h1_scaled, y2_scaled + h2_scaled) - y_intersection

    # Oblicz obszary bounding boxów i intersection
    area_box1 = w1_scaled * h1_scaled
    area_box2 = w2_scaled * h2_scaled
    area_intersection = max(0, w_intersection) * max(0, h_intersection)

    # Oblicz różnicę
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
    # Pobierz współczynniki skalowania
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]

    for position in ground_truth_positions:
        # Przeskaluj współrzędne bounding boxa
        x1, y1, x2, y2 = position['bbox']
        x1_scaled, y1_scaled, x2_scaled, y2_scaled = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

        # Narysuj przeskalowany bounding box
        cv.rectangle(image, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)

        # Przeskaluj etykietę
        label = f"{position['label']}: {position['alpha']:.2f}"
        cv.putText(image, label, (x1_scaled, y1_scaled - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


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
        decoded_predictions = decode_predictions(predicted_positions, top=3)[0]
        for i, prediction in enumerate(decoded_predictions):
            id, label, score = prediction
            print(f"{i + 1}: {label} ({score:.2f})")
        ground_truth_positions = read_ground_truth(image[0:6])

        # for gt_position in ground_truth_positions:
        #     draw_ground_truth(img, [gt_position])

        for pred_position in predicted_positions:
                for gt_position in ground_truth_positions:
                    diff = calculate_diff(pred_position, gt_position)
                # print(type(diff))
                print("Difference:", diff)
        
        # cv.imshow('Image with Ground Truth', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        if i == 1:
            break


if __name__ == '__main__':
    main()
