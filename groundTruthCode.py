import cv2
import tensorflow as tf
import os

# Funkcja do odczytu rzeczywistych położeń z pliku ground truth
def read_ground_truth(frame_number):
    file_path = f'/home/MaciejWozniakowski/programowanko/pythonProjects/PBL/training/label_2{frame_number:03d}.txt'
    
    # Tutaj dodaj kod do odczytu danych z pliku ground truth
    # Zwróć listę rzeczywistych położeń, na przykład [(x1, y1), (x2, y2), ...]
    # Możesz dostosować ten kod do formatu twoich danych w plikach ground truth.
    # Przykładowo, jeśli dane są zapisane w formie CSV, możesz użyć biblioteki csv.

    # Przykładowy kod (do dostosowania):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            ground_truth_positions = [(float(line.split(',')[0]), float(line.split(',')[1])) for line in lines]
            return ground_truth_positions
    else:
        return []

# Wczytaj model
model = tf.keras.applications.MobileNetV2(weights='/home/MaciejWozniakowski/programowanko/pythonProjects/PBL/YOLOv8-3D- mobilenetv2/mobilenetv2/mobilenetv2_weights.h5')


capture = cv2.VideoCapture('/home/MaciejWozniakowski/programowanko/pythonProjects/PBL/YOLOv8-3D- mobilenetv2/mobilenetv2_output_video.mp4')
while True:
    isTrue, frame = capture.read()
    if isTrue == False:
        break
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
    # Wykonaj predykcję na klatce
    predicted_positions = model.predict(frame)

    # Odczytaj rzeczywiste położenia z pliku ground truth
    ground_truth_positions = read_ground_truth(frame_number)

    # Porównaj predykcje z rzeczywistymi położeniami
    # i oblicz miary jakości

    # Zwiększ licznik numeru klatki

# Zamknij obiekt VideoCapture
cv2.destroyAllWindows()


