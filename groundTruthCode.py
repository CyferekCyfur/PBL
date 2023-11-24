import cv2 as cv
import tensorflow as tf
import os

# Function to calculate intersection of two boxes
def calculate_diff(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection coordinates
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection

    # Calculate areas of boxes and intersection
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    area_intersection = max(0, w_intersection) * max(0, h_intersection)

    # Calculate IoU
    diff = area_intersection / (area_box1 + area_box2 - area_intersection)
    return diff

# Function to read ground truth from a file
def read_ground_truth(frame_number):
    if os.path.exists(file_path):
        file_path = os.path.join(os.getcwd() + '/training/label_2/')
        with open(os.path.join(file_path + frame_number), 'r') as f:
            for line in f:
            # when we will know what the data in the file mean, we will read necessary data
             return ground_truth_positions
    else:
        return []

def main():
    # Read model
    model_path = os.path.join(os.getcwd() + '/YOLOV8-3D_models' + '/mobilenetv2/' + 'mobilenetv2_weights.h5')
    model = tf.keras.applications.MobileNetV2(weights=None)
    model.load_weights(model_path)

    video_path = os.path.join(os.getcwd() + '/YOLOV8-3D_models' + '/mobilenetv2/' + 'mobilenetv2_output_video.mp4')
    capture = cv.VideoCapture(video_path)
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break
        # Prediction on a frame
        predicted_positions = model.predict(tf.expand_dims(frame, axis=0)) #to be checked


        # Read real positions of objects
        frame_number = int(capture.get(cv.CAP_PROP_POS_FRAMES))
        ground_truth_positions = read_ground_truth(frame_number)

        # Compare predicted and real positions
        for pred_position in predicted_positions:
                for gt_position in ground_truth_positions:
                    diff = calculate_diff(pred_position, gt_position)
                    print("Difference:", diff)

    # Close the video
    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
