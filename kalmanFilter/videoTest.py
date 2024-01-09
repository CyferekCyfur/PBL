import cv2 as cv

# load videos into openCV
# capture = cv.VideoCapture("efficientnetb0_output_video.mp4")
capture = cv.VideoCapture("efficientnetb5_output_video.mp4")
desiredFrame = 200
# capture = cv.VideoCapture("mobilenetv2_output_video.mp4")
frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

# display frames one by one
# for frame in range(frames):
#    print(frame)
#    ret, img = capture.read()
#    if ret == False:
#        break
#    cv.imshow("Image", img)
#    if cv.waitKey(20) & 0xFF == ord("d"):  # key 'd' stops the video playback
#        break
framesList = []
for frame in range(frames):
    ret, img = capture.read()
    if ret == False:
        break
    framesList.append(img)


cv.imshow("img", framesList[100])
cv.waitKey()

capture.release()
cv.destroyAllWindows()
