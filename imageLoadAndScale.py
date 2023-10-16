import cv2 as cv
#load videos into openCV
capture1 = cv.VideoCapture('Vechicles/Dense/jan28.avi')
capture2 = cv.VideoCapture('Vechicles/Sunny/april21.avi')
capture3 = cv.VideoCapture('Vechicles/Urban/march9.avi')

#display frames one by one 
while True:
    isTrue, frame = capture3.read() # to change the video, just replace captureX with desired number, for example capture1
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'): #key 'd' stops the video playback
        break
capture.release()
cv.destroyAllWindows()