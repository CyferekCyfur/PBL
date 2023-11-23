import cv2
import tensorflow as tf
import os

# Funkcja do odczytu rzeczywistych położeń z pliku ground truth
def read_ground_truth(frame_number):
    file_path = os.path.join(os.getcwd() + 'training/label_2/')

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
model_path = os.path.join(os.getcwd() + '/mobilenetv2/' + 'mobilenetv2_weights.h5/')
model = tf.keras.applications.MobileNetV2(weights=model_path)

capture = cv2.VideoCapture(os.path.join(os.getcwd() + '/mobilenetv2/' + 'mobilenetv2_output_video.mp4'))
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


